# gpu-search-engine

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