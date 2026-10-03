import json
import os
import sqlite3
import sys
from pathlib import Path

import click
import tqdm
from google.protobuf import timestamp_pb2

# Dataset directories are standalone scripts, not Python packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import protocol_pb2
from util import download_file, get_embedding, read_jsonl, write_dump


BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
DIRECTORY = Path("lib/amazon-all-beauty")


def prepare(directory):
    stats_path = directory / "stats.json"
    if all((directory / name).exists() for name in ("stats.json", "users.jsonl", "items.jsonl")):
        return json.loads(stats_path.read_text(encoding="utf-8"))

    users = set()
    review_items = set()
    feedback_count = 0
    for review in tqdm.tqdm(read_jsonl(directory / "All_Beauty.jsonl.gz"), desc="Reading Reviews"):
        if not review["user_id"] or not review["parent_asin"]:
            raise ValueError("Review has an empty user_id or parent_asin")
        if not 1 <= float(review["rating"]) <= 5:
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
            read_jsonl(directory / "meta_All_Beauty.jsonl.gz"), desc="Preparing Items"
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
                [metadata.get("main_category") or "All Beauty"] + (metadata.get("categories") or [])
            ))
            labels = {key: metadata[key] for key in ("store", "price") if metadata.get(key) is not None}
            item = {"item_id": item_id, "title": title, "categories": categories,
                    "description": description, "labels": labels}
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
        # Keep feedback references valid even if metadata is unavailable.
        for item_id in sorted(review_items - metadata_items):
            output.write(json.dumps({
                "item_id": item_id, "title": "", "categories": ["All Beauty"],
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


def generate_embeddings(directory, database, total):
    database.execute("CREATE TABLE IF NOT EXISTS embeddings (item_id TEXT PRIMARY KEY, vector TEXT NOT NULL)")
    for item in tqdm.tqdm(read_jsonl(directory / "items.jsonl"), total=total, desc="Generating Embeddings"):
        if database.execute("SELECT 1 FROM embeddings WHERE item_id = ?", (item["item_id"],)).fetchone():
            continue
        embedding = get_embedding(item["description"]) if item["description"] else []
        database.execute("INSERT INTO embeddings VALUES (?, ?)", (item["item_id"], json.dumps(embedding)))
        # Save each response immediately, so interruptions do not repeat paid calls.
        database.commit()


def dump(directory, stats, database=None):
    path = directory / "amazon-all-beauty.bin"
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as output:
        output.write((-1).to_bytes(8, byteorder="little", signed=True))
        for user in tqdm.tqdm(read_jsonl(directory / "users.jsonl"), total=stats["users"], desc="Dumping Users"):
            write_dump(output, protocol_pb2.User(user_id=user["user_id"]))

        output.write((-2).to_bytes(8, byteorder="little", signed=True))
        for item in tqdm.tqdm(read_jsonl(directory / "items.jsonl"), total=stats["items"], desc="Dumping Items"):
            labels = dict(item["labels"])
            labels["description"] = item["description"]
            if database is not None:
                row = database.execute("SELECT vector FROM embeddings WHERE item_id = ?", (item["item_id"],)).fetchone()
                if row is None:
                    raise ValueError(f"Missing embedding for {item['item_id']}")
                labels["embedding"] = json.loads(row[0])
            write_dump(output, protocol_pb2.Item(
                item_id=item["item_id"], categories=item["categories"],
                comment=item["title"], labels=json.dumps(labels, ensure_ascii=False).encode("utf-8"),
            ))

        output.write((-3).to_bytes(8, byteorder="little", signed=True))
        for review in tqdm.tqdm(
            read_jsonl(directory / "All_Beauty.jsonl.gz"), total=stats["feedback"], desc="Dumping Feedback"
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


@click.command()
@click.option("--stop-before-embedding", is_flag=True, help="Stop after preparing users and product descriptions.")
@click.option("--skip-embedding", is_flag=True, help="Generate the dump without embedding API calls.")
def convert(stop_before_embedding, skip_embedding):
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    download_file(BASE_URL + "/review_categories/All_Beauty.jsonl.gz", DIRECTORY / "All_Beauty.jsonl.gz")
    download_file(BASE_URL + "/meta_categories/meta_All_Beauty.jsonl.gz", DIRECTORY / "meta_All_Beauty.jsonl.gz")
    stats = prepare(DIRECTORY)
    click.echo(json.dumps(stats, indent=2))
    if stop_before_embedding:
        return
    if skip_embedding:
        dump(DIRECTORY, stats)
    else:
        # Keep vectors on disk rather than loading the entire catalogue into RAM.
        with sqlite3.connect(DIRECTORY / "embeddings.sqlite") as database:
            generate_embeddings(DIRECTORY, database, stats["items"])
            dump(DIRECTORY, stats, database)


if __name__ == "__main__":
    convert()
