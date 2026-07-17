# gpu-search-engine

GPU-accelerated semantic search over ~50,000 BBC News articles, powered by a
custom [Metal](https://developer.apple.com/metal/) compute kernel that runs
the top-5 similarity search directly on the GPU (see `src/compute.metal`).

## Installing the CLI

```bash
npm install -g gpu-search-engine
gse "search query here"
```

Requirements: **macOS on Apple Silicon (arm64)**. The package ships a
precompiled Metal binary and does not build anything on install; there is no
Python or Xcode dependency at runtime. `package.json`'s `os`/`cpu` fields
will block installation on unsupported platforms.

Running `gse` with no arguments starts an interactive REPL (`Search: ` prompt,
Ctrl+C to exit); running `gse "<query>"` does a single one-shot search and
exits.

### How it works

1. `bin/gse.js` embeds the query text in Node using
   [`@huggingface/transformers`](https://www.npmjs.com/package/@huggingface/transformers)
   (the same `all-MiniLM-L6-v2` model used to build the corpus), entirely
   locally -- no network calls at query time.
2. The 384-dim query vector is streamed over stdin to a prebuilt native
   binary (`native/main-darwin-arm64`, compiled from `src/main.mm`), which
   loads the corpus's precomputed embeddings (`native/data/vector_db.bin`)
   once at startup and runs the similarity computation for each query on the
   GPU via the Metal kernel in `src/compute.metal`.
3. The binary returns the top-5 matching row indices over stdout; `gse.js`
   looks up the corresponding article text from `native/data/corpus.json`
   (index-aligned with `vector_db.bin`) and prints it.

## Development

The rest of this README documents the developer/maintainer workflow for
rebuilding the corpus and publishing new releases -- not needed to just use
the CLI.

## Refreshing the corpus and vector database

The corpus is sourced from the [`RealTimeData/bbc_news_alltime`](https://huggingface.co/datasets/RealTimeData/bbc_news_alltime) dataset via the shared `scripts/corpus.py` module. It exposes `get_corpus(max_rows=...)`, which loads Monthly_Config data newest-to-oldest and stops as soon as `max_rows` articles have been collected.

To rebuild the corpus and the Vector_Db_File (`data/vector_db.bin`), run:

```bash
python scripts/build_vector_db.py
```

This calls `corpus.get_corpus(max_rows=...)`, embeds each entry with the `sentence-transformers/all-MiniLM-L6-v2` model, and writes the resulting matrix to `data/vector_db.bin`.

`build_query_vector.py` (the interactive search client) loads the corpus the same way, via its own call to `corpus.get_corpus(max_rows=...)`. If you change how the corpus is built, re-run `build_query_vector.py` as well so the two stay in sync — the query client displays results by looking up the same corpus list by index, so a mismatched corpus will show the wrong article for a given result.

### Adjusting which Monthly_Config values are included

- **Total number of articles:** change the `max_rows` argument passed to `corpus.get_corpus(max_rows=...)` in `scripts/build_vector_db.py` and `build_query_vector.py` (and the `MAX_ROWS` constant at the top of `scripts/build_vector_db.py`). Loading always starts from the newest available month and works backwards, stopping once `max_rows` articles have been accumulated.
- **Newest-month starting point:** the starting point is whatever `corpus.list_available_months()` reports as the newest Monthly_Config for the dataset (config names are `YYYY-MM`, sorted newest-first). As new months are published upstream, re-running the build scripts will automatically pick them up first. There's no separate override for the starting month today — to exclude the very latest month(s), you'd need to filter the list returned by `list_available_months()` before passing it through, or lower `max_rows` so older months are still included proportionally less.

After changing either script, re-run `scripts/build_vector_db.py` to regenerate `data/vector_db.bin`, rebuild the search engine with `./build.sh` if `src/main.mm` changed, and re-run `build_query_vector.py` to query against the refreshed data.

`scripts/build_vector_db.py` also writes `data/corpus.json` (the same
`corpus_list`, index-aligned with `vector_db.bin`) for use by the npm CLI
package -- see "Publishing a new release" below.

## Publishing a new release

The npm package ships a prebuilt binary and a prebuilt corpus rather than
building either at install time, so publishing a new version requires an
explicit release step on a macOS Apple Silicon machine:

1. Rebuild the corpus and vector DB, if desired:
   ```bash
   python scripts/build_vector_db.py
   ```
   This also writes `data/corpus.json`, kept index-aligned with
   `data/vector_db.bin` automatically.
2. Assemble the package's native assets:
   ```bash
   ./release.sh
   ```
   This compiles `src/compute.metal` + `src/main.mm`, and copies the
   resulting binary, `default.metallib`, `data/vector_db.bin`, and
   `data/corpus.json` into `native/` -- the directory actually published in
   the npm package (see `package.json`'s `files` field).
3. Bump the version in `package.json`.
4. `npm publish`.

`native/` is gitignored; it's a build artifact regenerated by `release.sh`,
not checked into version control.