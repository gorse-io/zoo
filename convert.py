import click
import datetime
import json
import os
import protocol_pb2
import requests
import tqdm
import zipfile
from dotenv import load_dotenv
from google.protobuf import message, timestamp_pb2
from openai import OpenAI


load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE")
)


def get_description(title: str) -> str:
    prompt = f"Write a short description of the movie '{title}' in one paragraph."
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content


def get_embedding(text: str) -> list:
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return embedding.data[0].embedding


def write_dump(f, data: message.Message):
    bytes_data = data.SerializeToString()
    f.write(len(bytes_data).to_bytes(8, byteorder="little"))
    f.write(bytes_data)


@click.command()
@click.argument("dataset")
def convert(dataset):
    # Create lib directory if it doesn't exist
    os.makedirs("lib", exist_ok=True)

    if dataset == "ml-100k":
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        zip_path = "lib/ml-100k.zip"
        genres = [
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children's",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ]

        # Download the dataset
        if not os.path.exists(zip_path):
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get("content-length", 0))
            with open(zip_path, "wb") as f, tqdm.tqdm(
                desc="Downloading",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        # Extract the dataset
        if not os.path.exists("lib/ml-100k"):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                members = zip_ref.namelist()
                with tqdm.tqdm(
                    total=len(members), desc="Extracting", unit="file"
                ) as pbar:
                    for member in members:
                        zip_ref.extract(member, "lib")
                        pbar.update(1)

        # Generate descriptions
        if not os.path.exists("lib/ml-100k/u.description"):
            with open("lib/ml-100k/u.item", encoding="ISO-8859-1") as f:
                lines = f.readlines()
            for line in tqdm.tqdm(lines, desc="Generating Descriptions"):
                parts = line.split("|")
                movie_id = parts[0]
                title = parts[1]
                description = get_description(title)
                with open(
                    "lib/ml-100k/u.description", "a", encoding="utf-8"
                ) as desc_file:
                    desc_file.write(f"{movie_id}|{description}\n")

        # Generate embedding
        if not os.path.exists("lib/ml-100k/u.embedding"):
            with open("lib/ml-100k/u.description", encoding="utf-8") as f:
                lines = f.readlines()
            for line in tqdm.tqdm(lines, desc="Generating Embeddings"):
                parts = line.strip().split("|", 1)
                movie_id = parts[0]
                description = parts[1]
                embedding = get_embedding(description)
                with open(
                    "lib/ml-100k/u.embedding", "a", encoding="utf-8"
                ) as embed_file:
                    embed_file.write(f"{movie_id}|{'|'.join(map(str, embedding))}\n")

        with open("lib/ml-100k/ml-100k.bin", "wb") as f:
            # Dump users
            f.write((-1).to_bytes(8, byteorder="little", signed=True))
            with open("lib/ml-100k/u.user") as user_file:
                lines = user_file.readlines()
            for line in tqdm.tqdm(lines, desc="Dumping Users"):
                parts = line.strip().split("|")
                user_id = parts[0]
                labels = {
                    "age": int(parts[1]),
                    "gender": parts[2],
                    "occupation": parts[3],
                    "zip_code": parts[4],
                }
                write_dump(
                    f,
                    protocol_pb2.User(
                        user_id=user_id,
                        labels=json.dumps(labels).encode("utf-8"),
                    ),
                )
            # Dump items
            f.write((-2).to_bytes(8, byteorder="little", signed=True))
            embeddings = {}
            with open("lib/ml-100k/u.embedding", encoding="utf-8") as embed_file:
                for line in embed_file:
                    parts = line.strip().split("|")
                    item_id = parts[0]
                    embedding = list(map(float, parts[1:]))
                    embeddings[item_id] = embedding
            with open("lib/ml-100k/u.item", encoding="ISO-8859-1") as item_file:
                lines = item_file.readlines()
            for line in tqdm.tqdm(lines, desc="Dumping Items"):
                parts = line.strip().split("|")
                item_id = parts[0]
                title = parts[1]
                release_date = parts[2]
                categories = [g for i, g in enumerate(genres) if parts[5 + i] == "1"]
                labels = {
                    "embedding": embeddings.get(item_id, []),
                }
                timestamp_pb = timestamp_pb2.Timestamp()
                if release_date:
                    timestamp_pb.FromDatetime(
                        datetime.datetime.strptime(release_date, "%d-%b-%Y")
                    )
                write_dump(
                    f,
                    protocol_pb2.Item(
                        item_id=item_id,
                        categories=categories,
                        timestamp=timestamp_pb,
                        labels=json.dumps(labels).encode("utf-8"),
                        comment=title,
                    ),
                )
            # Dump feedback
            f.write((-3).to_bytes(8, byteorder="little", signed=True))
            with open("lib/ml-100k/u.data") as data_file:
                lines = data_file.readlines()
            for line in tqdm.tqdm(lines, desc="Dumping Feedback"):
                parts = line.strip().split("\t")
                user_id = parts[0]
                item_id = parts[1]
                rating = float(parts[2])
                timestamp = int(parts[3])
                timestamp_pb = timestamp_pb2.Timestamp()
                timestamp_pb.FromSeconds(timestamp)
                write_dump(
                    f,
                    protocol_pb2.Feedback(
                        feedback_type="rating",
                        user_id=user_id,
                        item_id=item_id,
                        value=rating,
                        timestamp=timestamp_pb,
                    ),
                )
            f.write((0).to_bytes(8, byteorder="little", signed=True))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


if __name__ == "__main__":
    convert()
