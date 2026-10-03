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
    temporary_directory, binary_directory = dataset_directories("ml-1m")
    directory = str(temporary_directory / "ml-1m")
    zip_path = str(temporary_directory / "ml-1m.zip")
    output_path = binary_directory / "ml-1m.bin"
    temporary_dump = temporary_directory / "ml-1m.bin.part"
    if not os.path.exists(directory):
        if not os.path.exists(zip_path):
            url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with open(zip_path + ".part", "wb") as output, tqdm.tqdm(
                    desc="Downloading", total=int(response.headers.get("content-length", 0)),
                    unit="B", unit_scale=True,
                ) as progress:
                    for chunk in response.iter_content(chunk_size=8192):
                        output.write(chunk)
                        progress.update(len(chunk))
            os.replace(zip_path + ".part", zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            # Extract only the expected dataset files.
            for name in ("movies.dat", "users.dat", "ratings.dat", "README"):
                archive.extract("ml-1m/" + name, temporary_directory)

    with open(directory + "/movies.dat", encoding="ISO-8859-1") as source:
        movies = [line.rstrip("\r\n").split("::", 2) for line in source]

    # Generate descriptions. Resume individual movies after an interrupted run.
    description_path = directory + "/movies.description"
    descriptions = {}
    if os.path.exists(description_path):
        with open(description_path, encoding="utf-8") as source:
            for line in source:
                movie_id, description = line.rstrip("\r\n").split("|", 1)
                if description.strip():
                    descriptions[movie_id] = description
    for movie_id, title, _ in tqdm.tqdm(movies, desc="Generating Descriptions"):
        if movie_id in descriptions:
            continue
        description = " ".join(get_description(title).split())
        if not description:
            raise ValueError(f"Empty description for movie {movie_id}")
        with open(description_path, "a", encoding="utf-8") as output:
            output.write(f"{movie_id}|{description}\n")
        descriptions[movie_id] = description

    if stop_before_embedding:
        return

    # Generate embedding
    embedding_path = directory + "/movies.embedding"
    embeddings = {}
    if os.path.exists(embedding_path):
        with open(embedding_path, encoding="utf-8") as source:
            for line in source:
                parts = line.rstrip("\r\n").split("|")
                embeddings[parts[0]] = list(map(float, parts[1:]))
    for movie_id, _, _ in tqdm.tqdm(movies, desc="Generating Embeddings"):
        if movie_id in embeddings:
            continue
        embedding = get_embedding(descriptions[movie_id])
        with open(embedding_path, "a", encoding="utf-8") as output:
            output.write(f"{movie_id}|{'|'.join(map(str, embedding))}\n")
        embeddings[movie_id] = embedding

    occupations = [
        "other", "academic/educator", "artist", "clerical/admin",
        "college/grad student", "customer service", "doctor/health care",
        "executive/managerial", "farmer", "homemaker", "K-12 student",
        "lawyer", "programmer", "retired", "sales/marketing", "scientist",
        "self-employed", "technician/engineer", "tradesman/craftsman",
        "unemployed", "writer",
    ]
    with open(temporary_dump, "wb") as output:
        # Dump users. The age field is the dataset's age-group code.
        output.write((-1).to_bytes(8, byteorder="little", signed=True))
        with open(directory + "/users.dat", encoding="ISO-8859-1") as source:
            for line in tqdm.tqdm(source, desc="Dumping Users"):
                user_id, gender, age, occupation, zip_code = line.rstrip("\r\n").split("::")
                labels = {"gender": gender, "age": int(age),
                          "occupation": occupations[int(occupation)], "zip_code": zip_code}
                write_dump(output, protocol_pb2.User(
                    user_id=user_id, labels=json.dumps(labels).encode("utf-8"),
                ))
        # Dump items. MovieLens 1M supplies only a release year.
        output.write((-2).to_bytes(8, byteorder="little", signed=True))
        for movie_id, title, genres in tqdm.tqdm(movies, desc="Dumping Items"):
            timestamp = timestamp_pb2.Timestamp()
            year = title.rsplit("(", 1)[-1].rstrip(")")
            if len(year) == 4 and year.isdigit():
                timestamp.FromDatetime(datetime.datetime(int(year), 1, 1))
            write_dump(output, protocol_pb2.Item(
                item_id=movie_id, categories=genres.split("|"), timestamp=timestamp,
                labels=json.dumps({"embedding": embeddings[movie_id]}).encode("utf-8"),
                comment=title,
            ))
        # Dump feedback without loading one million ratings into memory.
        output.write((-3).to_bytes(8, byteorder="little", signed=True))
        with open(directory + "/ratings.dat", encoding="ISO-8859-1") as source:
            for line in tqdm.tqdm(source, desc="Dumping Feedback"):
                user_id, movie_id, rating, seconds = line.rstrip("\r\n").split("::")
                timestamp = timestamp_pb2.Timestamp()
                timestamp.FromSeconds(int(seconds))
                write_dump(output, protocol_pb2.Feedback(
                    feedback_type="rating", user_id=user_id, item_id=movie_id,
                    value=float(rating), timestamp=timestamp,
                ))
        output.write((0).to_bytes(8, byteorder="little", signed=True))
    os.replace(temporary_dump, output_path)

if __name__ == "__main__":
    convert()
