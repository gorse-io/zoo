# Recommender systems datasets for Gorse

A collection of datasets in Gorse dump format.

| Dataset | #Users | #Items | #Feedback |
|---------|--------|--------|-----------|
| [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) | 943 | 1,682 | 100,000 |
| [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) | 6,040 | 3,883 | 1,000,209 |

## Dataset scripts

Each dataset has its own standalone script. Shared API and dump-writing helpers
live in the root `util.py`. Run commands from the repository root; downloaded
files and generated artifacts are stored under `lib/<dataset>/`.

```sh
python ml-100k/convert.py --stop-before-embedding
python ml-1m/convert.py --stop-before-embedding
```

Omit `--stop-before-embedding` to generate embeddings and the binary dump.

## Convert MovieLens 1M

Install the dependencies with `pip install -r requirements.txt`. Configure
`OPENAI_API_KEY` and, optionally, `OPENAI_API_BASE` and `OPENAI_MODEL` in the
environment or a local `.env` file. The default description model is
`gpt-4o-mini`; an OpenAI-compatible description service can be used by setting
the base URL and model (for example, `deepseek-v4-flash`).

Download, extract, and generate movie descriptions, stopping before embeddings:

```sh
python ml-1m/convert.py --stop-before-embedding
```

Descriptions are cached in `lib/ml-1m/movies.description`. Re-running resumes
missing movies. No embeddings or binary dump are generated with this option.

To continue with embeddings and produce `lib/ml-1m/ml-1m.bin`:

```sh
python ml-1m/convert.py
```

The embedding step requires a service supporting `text-embedding-3-small`;
configure `OPENAI_API_BASE` and `OPENAI_API_KEY` for that service before continuing.
The dump preserves the original age-group codes, converts occupation codes to
names, and uses January 1 of the release year for item timestamps because this
dataset does not include exact release dates.

Run conversion tests (LLM responses are mocked; no paid API calls):

```sh
python -m unittest -v test_convert
```
