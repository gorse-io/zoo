# Recommender systems datasets for Gorse

A collection of datasets in Gorse dump format.

| Dataset | #Users | #Items | #Feedback |
|---------|--------|--------|-----------|
| [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) | 943 | 1,682 | 100,000 |
| [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) | 6,040 | 3,883 | 1,000,209 |
| [Amazon All Beauty 2023](https://amazon-reviews-2023.github.io/) | 631,986 | 112,590 | 701,528 |

## Dataset scripts

Each dataset has its own standalone script. Shared API and dump-writing helpers
live in the root `util.py`. Run the example commands from the repository root.
Downloads, prepared data, descriptions, embeddings, and temporary `.part` files
are stored in each dataset's `tmp/` directory. Complete dumps are stored in its
`bin/` directory. Paths are resolved relative to the repository, not the current
working directory.

```text
amazon-all-beauty/
├── convert.py
├── tmp/    # downloads, prepared data, embedding cache, incomplete dumps
└── bin/    # amazon-all-beauty.bin
```

Both `tmp/` and `bin/` are ignored by Git. A dump is first written to
`tmp/<dataset>.bin.part` and moved into `bin/<dataset>.bin` only after it is
complete. Interrupted output therefore does not replace a previous complete dump.

MovieLens ZIP archives are stored in `ml-100k/tmp/` or `ml-1m/tmp/`; their
extracted source files and description/embedding caches are stored in
`ml-100k/tmp/ml-100k/` or `ml-1m/tmp/ml-1m/`.
Old `lib/` paths are no longer read; move existing caches to the new directories
before re-running if you want to avoid regenerating descriptions or embeddings.

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

Descriptions are cached in `ml-1m/tmp/ml-1m/movies.description`. Re-running resumes
missing movies. No embeddings or binary dump are generated with this option.

To continue with embeddings and produce `ml-1m/bin/ml-1m.bin`:

```sh
python ml-1m/convert.py
```

The embedding step requires a service supporting `text-embedding-3-small`;
configure `OPENAI_API_BASE` and `OPENAI_API_KEY` for that service before continuing.
The dump preserves the original age-group codes, converts occupation codes to
names, and uses January 1 of the release year for item timestamps because this
dataset does not include exact release dates.

## Convert Amazon Reviews 2023 datasets

Each category has a standalone script; downloading, preparing users and items,
embedding caching, and dump generation are shared in `util.py`. Each converter
downloads its category's raw reviews and product metadata from the official
Amazon Reviews 2023 dataset. It reads gzip JSONL
files directly, without unpacking them. Product IDs use `parent_asin` rather
than variant-level `asin`, matching the metadata.

Prepare users and products without any API calls:

```sh
python amazon-all-beauty/convert.py --stop-before-embedding
```

Descriptions use the existing product title, description, and features; no LLM
text generation is needed. Prepared data and counts are cached as
`amazon-all-beauty/tmp/users.jsonl`, `items.jsonl`, and `stats.json`.

Generate a complete dump without embeddings:

```sh
python amazon-all-beauty/convert.py --skip-embedding
```

To generate embeddings and then the dump, configure `OPENAI_API_KEY` and an
optional `OPENAI_API_BASE` supporting `text-embedding-3-small`, then run:

```sh
python amazon-all-beauty/convert.py
```

Embedding responses are cached per product in `tmp/embeddings.sqlite`, allowing
interrupted runs to resume without keeping all vectors in memory. Products
without any text do not make an embedding request and have an empty vector.

The output is `amazon-all-beauty/bin/amazon-all-beauty.bin`. Product labels retain
`description`, `store`, and `price` when available; embeddings are added unless
`--skip-embedding` is used. Comments preserve product titles and review text.
Review ratings are exported as `rating` feedback with millisecond timestamps.
All raw reviews are retained in source order, including repeated user/product
pairs; the feedback count is the number of exported records, not unique pairs.

For `All_Beauty`, the raw metadata contains 112,590 products, including 25 with
no reviews; the converter retains all of them. There are 112,565 products
referenced by reviews.
If a reviewed product lacks metadata, it is retained with a
`metadata_missing` label rather than dropping its feedback. Item timestamps
are unset because the metadata does not supply product release dates.

The source dataset's usage terms apply to the downloaded files. Only conversion
code is tracked in this repository, not the source data or generated dumps.

### Available Amazon category scripts

All category scripts use the same options shown above. For example:

```sh
python amazon-books/convert.py --stop-before-embedding
python amazon-electronics/convert.py --skip-embedding
```

Inputs and intermediate caches are stored under `<dataset>/tmp/`, and the dump
is `<dataset>/bin/<dataset>.bin`.
The additional scripts have not been run; no additional datasets were downloaded
or converted. Large categories require enough disk space for the compressed
sources and generated artifacts, plus RAM for the user and product ID sets.

| Source category | Script |
|-----------------|--------|
| `All_Beauty` | `amazon-all-beauty/convert.py` |
| `Amazon_Fashion` | `amazon-fashion/convert.py` |
| `Appliances` | `amazon-appliances/convert.py` |
| `Arts_Crafts_and_Sewing` | `amazon-arts-crafts-and-sewing/convert.py` |
| `Automotive` | `amazon-automotive/convert.py` |
| `Baby_Products` | `amazon-baby-products/convert.py` |
| `Beauty_and_Personal_Care` | `amazon-beauty-and-personal-care/convert.py` |
| `Books` | `amazon-books/convert.py` |
| `CDs_and_Vinyl` | `amazon-cds-and-vinyl/convert.py` |
| `Cell_Phones_and_Accessories` | `amazon-cell-phones-and-accessories/convert.py` |
| `Clothing_Shoes_and_Jewelry` | `amazon-clothing-shoes-and-jewelry/convert.py` |
| `Digital_Music` | `amazon-digital-music/convert.py` |
| `Electronics` | `amazon-electronics/convert.py` |
| `Gift_Cards` | `amazon-gift-cards/convert.py` |
| `Grocery_and_Gourmet_Food` | `amazon-grocery-and-gourmet-food/convert.py` |
| `Handmade_Products` | `amazon-handmade-products/convert.py` |
| `Health_and_Household` | `amazon-health-and-household/convert.py` |
| `Health_and_Personal_Care` | `amazon-health-and-personal-care/convert.py` |
| `Home_and_Kitchen` | `amazon-home-and-kitchen/convert.py` |
| `Industrial_and_Scientific` | `amazon-industrial-and-scientific/convert.py` |
| `Kindle_Store` | `amazon-kindle-store/convert.py` |
| `Magazine_Subscriptions` | `amazon-magazine-subscriptions/convert.py` |
| `Movies_and_TV` | `amazon-movies-and-tv/convert.py` |
| `Musical_Instruments` | `amazon-musical-instruments/convert.py` |
| `Office_Products` | `amazon-office-products/convert.py` |
| `Patio_Lawn_and_Garden` | `amazon-patio-lawn-and-garden/convert.py` |
| `Pet_Supplies` | `amazon-pet-supplies/convert.py` |
| `Software` | `amazon-software/convert.py` |
| `Sports_and_Outdoors` | `amazon-sports-and-outdoors/convert.py` |
| `Subscription_Boxes` | `amazon-subscription-boxes/convert.py` |
| `Tools_and_Home_Improvement` | `amazon-tools-and-home-improvement/convert.py` |
| `Toys_and_Games` | `amazon-toys-and-games/convert.py` |
| `Video_Games` | `amazon-video-games/convert.py` |
| `Unknown` | `amazon-unknown/convert.py` |
