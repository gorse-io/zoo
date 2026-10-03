# Recommender systems datasets for Gorse

A collection of datasets in Gorse dump format.

| Dataset | #Users | #Items | #Feedback |
|---------|--------|--------|-----------|
| [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) | 943 | 1,682 | 100,000 |
| [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) | 6,040 | 3,883 | 1,000,209 |
| [Amazon All Beauty 2023](https://amazon-reviews-2023.github.io/) | 631,986 | 112,590 | 701,528 |

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

## Convert Amazon All Beauty 2023

The standalone converter downloads the raw `All_Beauty` reviews and product
metadata from the official Amazon Reviews 2023 dataset. It reads gzip JSONL
files directly, without unpacking them. Product IDs use `parent_asin` rather
than variant-level `asin`, matching the metadata.

Prepare users and products without any API calls:

```sh
python amazon-all-beauty/convert.py --stop-before-embedding
```

Descriptions use the existing product title, description, and features; no LLM
text generation is needed. Prepared data and counts are cached as
`lib/amazon-all-beauty/users.jsonl`, `items.jsonl`, and `stats.json`.

Generate a complete dump without embeddings:

```sh
python amazon-all-beauty/convert.py --skip-embedding
```

To generate embeddings and then the dump, configure `OPENAI_API_KEY` and an
optional `OPENAI_API_BASE` supporting `text-embedding-3-small`, then run:

```sh
python amazon-all-beauty/convert.py
```

Embedding responses are cached per product in `embeddings.sqlite`, allowing
interrupted runs to resume without keeping all vectors in memory. Products
without any text do not make an embedding request and have an empty vector.

The output is `lib/amazon-all-beauty/amazon-all-beauty.bin`. Product labels retain
`description`, `store`, and `price` when available; embeddings are added unless
`--skip-embedding` is used. Comments preserve product titles and review text.
Review ratings are exported as `rating` feedback with millisecond timestamps.
All raw reviews are retained in source order, including repeated user/product
pairs; the feedback count is the number of exported records, not unique pairs.

The raw metadata contains 112,590 products, including 25 with no reviews; the
converter retains all of them. There are 112,565 products referenced by reviews.
If a reviewed product lacks metadata, it is retained with a
`metadata_missing` label rather than dropping its feedback. Item timestamps
are unset because the metadata does not supply product release dates.

The source dataset's usage terms apply to the downloaded files. Only conversion
code is tracked in this repository, not the source data or generated dumps.
