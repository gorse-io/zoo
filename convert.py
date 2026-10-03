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
@click.option("--stop-before-embedding", is_flag=True, help="Stop after generating movie descriptions.")
def convert(dataset, stop_before_embedding):
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

        if stop_before_embedding:
            return

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
    elif dataset == "ml-1m":
        convert_ml1m(stop_before_embedding)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def convert_ml1m(stop_before_embedding):
    directory = "lib/ml-1m"
    zip_path = "lib/ml-1m.zip"
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
                archive.extract("ml-1m/" + name, "lib")

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
    with open(directory + "/ml-1m.bin", "wb") as output:
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


if __name__ == "__main__":
    convert()
