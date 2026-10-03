import os

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
