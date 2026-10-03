import gzip
import json
import os
from pathlib import Path

import requests
import tqdm

from dotenv import load_dotenv
from google.protobuf import message
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
