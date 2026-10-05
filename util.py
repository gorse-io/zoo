import gzip
import json
import os
import sqlite3
import struct
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import click
import protocol_pb2
import requests
import tqdm

from dotenv import load_dotenv
from google.protobuf import message, timestamp_pb2
from openai import OpenAI


load_dotenv()


client = None


def get_client():
    global client
    if client is None:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE")
        )
    return client


def get_description(title: str) -> str:
    prompt = f"Write a short description of the movie '{title}' in one paragraph."
    completion = get_client().chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content


def get_embedding(text: str) -> list:
    embedding = get_client().embeddings.create(
        model="qwen3.7-text-embedding-flash",
        input=text,
    )
    return embedding.data[0].embedding


def write_dump(f, data: message.Message):
    bytes_data = data.SerializeToString()
    f.write(len(bytes_data).to_bytes(8, byteorder="little"))
    f.write(bytes_data)


def dataset_directories(dataset):
    """Return dataset-local intermediate and final-output directories."""
    directory = Path(__file__).resolve().parent / dataset
    temporary = directory / "tmp"
    binary = directory / "bin"
    temporary.mkdir(parents=True, exist_ok=True)
    binary.mkdir(parents=True, exist_ok=True)
    return temporary, binary


def download_file(url, path):
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with temporary.open("wb") as output, tqdm.tqdm(
            desc=f"Downloading {path.name}",
            total=int(response.headers.get("content-length", 0)),
            unit="B", unit_scale=True,
        ) as progress:
            for chunk in response.iter_content(chunk_size=8192):
                output.write(chunk)
                progress.update(len(chunk))
    os.replace(temporary, path)


def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def prepare_amazon(directory, category):
    stats_path = directory / "stats.json"
    if all((directory / name).exists() for name in ("stats.json", "users.jsonl", "items.jsonl")):
        return json.loads(stats_path.read_text(encoding="utf-8"))

    users = set()
    review_items = set()
    feedback_count = 0
    for review in tqdm.tqdm(read_jsonl(directory / f"{category}.jsonl.gz"), desc="Reading Reviews"):
        if not review["user_id"] or not review["parent_asin"]:
            raise ValueError("Review has an empty user_id or parent_asin")
        # Some source categories contain zero ratings; preserve them verbatim.
        if not 0 <= float(review["rating"]) <= 5:
            raise ValueError(f"Invalid rating: {review['rating']}")
        # Check timestamp range before producing any prepared artifacts.
        timestamp_pb2.Timestamp().FromMilliseconds(int(review["timestamp"]))
        users.add(review["user_id"])
        review_items.add(review["parent_asin"])
        feedback_count += 1

    with (directory / "users.jsonl.part").open("w", encoding="utf-8") as output:
        for user_id in sorted(users):
            output.write(json.dumps({"user_id": user_id}) + "\n")
    os.replace(directory / "users.jsonl.part", directory / "users.jsonl")

    metadata_items = set()
    with (directory / "items.jsonl.part").open("w", encoding="utf-8") as output:
        for metadata in tqdm.tqdm(
            read_jsonl(directory / f"meta_{category}.jsonl.gz"), desc="Preparing Items"
        ):
            item_id = metadata["parent_asin"]
            if not item_id:
                raise ValueError("Metadata has an empty parent_asin")
            if item_id in metadata_items:
                raise ValueError(f"Duplicate metadata for {item_id}")
            metadata_items.add(item_id)
            title = (metadata.get("title") or "").strip()
            # Use existing product text; no movie-description LLM is needed.
            text = [title] + (metadata.get("description") or []) + (metadata.get("features") or [])
            description = "\n".join(dict.fromkeys(part.strip() for part in text if part.strip()))
            categories = list(dict.fromkeys(
                [metadata.get("main_category") or category.replace("_", " ")] + (metadata.get("categories") or [])
            ))
            labels = {
                key: metadata[key]
                for key in ("store", "average_rating", "rating_number", "price")
                if metadata.get(key) is not None
            }
            item = {"item_id": item_id, "title": title, "categories": categories,
                    "description": description, "labels": labels}
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
        # Keep feedback references valid even if metadata is unavailable.
        for item_id in sorted(review_items - metadata_items):
            output.write(json.dumps({
                "item_id": item_id, "title": "", "categories": [category.replace("_", " ")],
                "description": "", "labels": {"metadata_missing": True},
            }) + "\n")
    os.replace(directory / "items.jsonl.part", directory / "items.jsonl")

    stats = {
        "users": len(users), "items": len(review_items | metadata_items),
        "feedback": feedback_count, "review_items": len(review_items),
        "metadata_items": len(metadata_items),
        "items_missing_metadata": len(review_items - metadata_items),
        "metadata_only_items": len(metadata_items - review_items),
    }
    (directory / "stats.json.part").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    os.replace(directory / "stats.json.part", stats_path)
    return stats


def generate_amazon_embeddings(directory, database, total, workers=1):
    # WAL avoids rewriting the rollback journal for every cached API response.
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("PRAGMA synchronous=NORMAL")
    database.execute("CREATE TABLE IF NOT EXISTS embeddings (item_id TEXT PRIMARY KEY, vector BLOB NOT NULL)")
    cached = {row[0] for row in database.execute("SELECT item_id FROM embeddings")}
    items = (item for item in read_jsonl(directory / "items.jsonl") if item["item_id"] not in cached)
    # Initialize the shared HTTP client before worker threads start.
    get_client()
    with ThreadPoolExecutor(max_workers=workers) as executor, tqdm.tqdm(
        total=total, initial=len(cached), desc="Generating Embeddings"
    ) as progress:
        pending = {}
        failure = None

        def submit_next():
            item = next(items, None)
            if item is None:
                return False
            future = executor.submit(get_embedding, item["description"]) if item["description"] else executor.submit(list)
            pending[future] = item["item_id"]
            return True

        for _ in range(workers):
            if not submit_next():
                break
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                item_id = pending.pop(future)
                try:
                    embedding = future.result()
                except Exception as error:
                    failure = failure or error
                    continue
                vector = struct.pack(f"<{len(embedding)}f", *embedding)
                # Only the main thread accesses SQLite; persist each successful response.
                database.execute("INSERT INTO embeddings VALUES (?, ?)", (item_id, vector))
                database.commit()
                progress.update(1)
                if failure is None:
                    submit_next()
        if failure is not None:
            raise failure


def dump_amazon(directory, category, stats, output_path, database=None):
    path = Path(output_path)
    temporary = directory / (path.name + ".part")
    with temporary.open("wb") as output:
        output.write((-1).to_bytes(8, byteorder="little", signed=True))
        for user in tqdm.tqdm(read_jsonl(directory / "users.jsonl"), total=stats["users"], desc="Dumping Users"):
            write_dump(output, protocol_pb2.User(user_id=user["user_id"]))

        output.write((-2).to_bytes(8, byteorder="little", signed=True))
        for item in tqdm.tqdm(read_jsonl(directory / "items.jsonl"), total=stats["items"], desc="Dumping Items"):
            labels = dict(item["labels"])
            if database is not None:
                row = database.execute("SELECT vector FROM embeddings WHERE item_id = ?", (item["item_id"],)).fetchone()
                if row is None:
                    raise ValueError(f"Missing embedding for {item['item_id']}")
                labels["embedding"] = list(struct.unpack(f"<{len(row[0]) // 4}f", row[0]))
            write_dump(output, protocol_pb2.Item(
                item_id=item["item_id"], categories=item["categories"],
                comment=item["title"], labels=json.dumps(labels, ensure_ascii=False).encode("utf-8"),
            ))

        output.write((-3).to_bytes(8, byteorder="little", signed=True))
        for review in tqdm.tqdm(
            read_jsonl(directory / f"{category}.jsonl.gz"), total=stats["feedback"], desc="Dumping Feedback"
        ):
            timestamp = timestamp_pb2.Timestamp()
            timestamp.FromMilliseconds(int(review["timestamp"]))
            comment = "\n".join(text for text in (review.get("title"), review.get("text")) if text)
            write_dump(output, protocol_pb2.Feedback(
                feedback_type="rating", user_id=review["user_id"], item_id=review["parent_asin"],
                value=float(review["rating"]), timestamp=timestamp, comment=comment,
            ))
        output.write((0).to_bytes(8, byteorder="little", signed=True))
    os.replace(temporary, path)
    click.echo(f"Dump: {path}")


AMAZON_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"


def amazon_converter(category, dataset):
    """Build a standalone CLI for one Amazon Reviews 2023 category."""
    @click.command()
    @click.option("--stop-before-embedding", is_flag=True, help="Stop after preparing users and product descriptions.")
    @click.option("--skip-embedding", is_flag=True, help="Generate the dump without embedding API calls.")
    @click.option("--embedding-workers", type=click.IntRange(min=1), default=1, show_default=True,
                  help="Maximum concurrent embedding requests.")
    def convert(stop_before_embedding, skip_embedding, embedding_workers):
        directory, binary_directory = dataset_directories(dataset)
        output_path = binary_directory / f"{dataset}.bin"
        review_name = f"{category}.jsonl.gz"
        metadata_name = f"meta_{category}.jsonl.gz"
        download_file(f"{AMAZON_BASE_URL}/review_categories/{review_name}", directory / review_name)
        download_file(f"{AMAZON_BASE_URL}/meta_categories/{metadata_name}", directory / metadata_name)
        stats = prepare_amazon(directory, category)
        click.echo(json.dumps(stats, indent=2))
        if stop_before_embedding:
            return
        if skip_embedding:
            dump_amazon(directory, category, stats, output_path)
        else:
            # Keep vectors on disk rather than loading the entire catalogue into RAM.
            with sqlite3.connect(directory / "embeddings.sqlite") as database:
                generate_amazon_embeddings(directory, database, stats["items"], embedding_workers)
                dump_amazon(directory, category, stats, output_path, database)

    return convert
