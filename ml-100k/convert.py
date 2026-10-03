import datetime
import json
import os
import sys
import zipfile
from pathlib import Path

import click
import requests
import tqdm
from google.protobuf import timestamp_pb2

# Allow direct execution without treating dataset directories as Python packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import protocol_pb2
from util import dataset_directories, get_description, get_embedding, write_dump


@click.command()
@click.option("--stop-before-embedding", is_flag=True, help="Stop after generating movie descriptions.")
def convert(stop_before_embedding):
    temporary_directory, binary_directory = dataset_directories("ml-100k")
    directory = str(temporary_directory / "ml-100k")
    output_path = binary_directory / "ml-100k.bin"
    temporary_dump = temporary_directory / "ml-100k.bin.part"

    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = str(temporary_directory / "ml-100k.zip")
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
    if not os.path.exists(directory):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            members = zip_ref.namelist()
            with tqdm.tqdm(
                total=len(members), desc="Extracting", unit="file"
            ) as pbar:
                for member in members:
                    zip_ref.extract(member, temporary_directory)
                    pbar.update(1)

    # Generate descriptions
    if not os.path.exists(directory + "/u.description"):
        with open(directory + "/u.item", encoding="ISO-8859-1") as f:
            lines = f.readlines()
        for line in tqdm.tqdm(lines, desc="Generating Descriptions"):
            parts = line.split("|")
            movie_id = parts[0]
            title = parts[1]
            description = get_description(title)
            with open(
                directory + "/u.description", "a", encoding="utf-8"
            ) as desc_file:
                desc_file.write(f"{movie_id}|{description}\n")

    if stop_before_embedding:
        return

    # Generate embedding
    if not os.path.exists(directory + "/u.embedding"):
        with open(directory + "/u.description", encoding="utf-8") as f:
            lines = f.readlines()
        for line in tqdm.tqdm(lines, desc="Generating Embeddings"):
            parts = line.strip().split("|", 1)
            movie_id = parts[0]
            description = parts[1]
            embedding = get_embedding(description)
            with open(
                directory + "/u.embedding", "a", encoding="utf-8"
            ) as embed_file:
                embed_file.write(f"{movie_id}|{'|'.join(map(str, embedding))}\n")

    with open(temporary_dump, "wb") as f:
        # Dump users
        f.write((-1).to_bytes(8, byteorder="little", signed=True))
        with open(directory + "/u.user") as user_file:
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
        with open(directory + "/u.embedding", encoding="utf-8") as embed_file:
            for line in embed_file:
                parts = line.strip().split("|")
                item_id = parts[0]
                embedding = list(map(float, parts[1:]))
                embeddings[item_id] = embedding
        with open(directory + "/u.item", encoding="ISO-8859-1") as item_file:
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
        with open(directory + "/u.data") as data_file:
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
    os.replace(temporary_dump, output_path)

if __name__ == "__main__":
    convert()
