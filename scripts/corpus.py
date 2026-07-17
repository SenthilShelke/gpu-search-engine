import time

from datasets import get_dataset_config_names, load_dataset
from dotenv import load_dotenv
from huggingface_hub.errors import HfHubHTTPError

load_dotenv()

DATASET_NAME = "RealTimeData/bbc_news_alltime"


class CorpusLoadError(Exception):
    """Raised when a Monthly_Config fails to load. str(err) includes the month name."""

    def __init__(self, month: str):
        super().__init__(f"Failed to load Monthly_Config '{month}'")
        self.month = month


def combine_title_description(title: str, description: str) -> str | None:
    """
    Pure function implementing Requirement 2.
    Returns None if title is empty/whitespace-only (article excluded).
    Otherwise returns title, or title + " " + description if description is non-empty.
    """
    if title is None or title.strip() == "":
        return None

    if description is None or description.strip() == "":
        return title

    return f"{title} {description}"


def list_available_months() -> list[str]:
    """
    Return all Monthly_Config names available for RealTimeData/bbc_news_alltime,
    sorted newest-first (descending, e.g. ["2025-06", "2025-05", ..., "2017-01"]).
    Uses huggingface_hub / datasets config discovery. Config names are of the
    form "YYYY-MM", so a lexicographic descending sort is chronologically correct.
    """
    months = get_dataset_config_names(DATASET_NAME)
    return sorted(months, reverse=True)


def _default_month_loader(month: str, max_retries: int = 5):
    """
    Default per-month loader used by get_corpus. Loads the Monthly_Config's
    records from the RealTimeData/bbc_news_alltime dataset via `datasets`.

    This is a thin wrapper around `datasets.load_dataset` so that `get_corpus`
    can accept an injectable loader function (defaulting to this one) and
    tests can substitute a fake loader instead of hitting the network.

    Retries with exponential backoff on HF Hub rate-limit (429) errors, since
    unauthenticated (or even authenticated) requests can hit the Hub's API
    rate limit when resolving many Monthly_Config file listings in a row.
    """
    for attempt in range(max_retries):
        try:
            return load_dataset(DATASET_NAME, month, split="train")
        except HfHubHTTPError as exc:
            response = getattr(exc, "response", None)
            is_rate_limit = response is not None and response.status_code == 429
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)


def get_corpus(max_rows: int = 50000, month_loader=_default_month_loader) -> list[str]:
    """
    Public entry point used by both scripts (Requirement 5).

    Loads Monthly_Config data newest-to-oldest via list_available_months(),
    applies combine_title_description() to each article, appends non-None
    results to an accumulator, and stops as soon as len(accumulator) >= max_rows.
    Truncates the final list to exactly max_rows if it overshot mid-month.
    Deterministic: same max_rows always yields the same list (Requirement 4).

    `month_loader` is an injectable per-month loader function (month: str) ->
    iterable of records, defaulting to `_default_month_loader`. Tests can
    substitute a fake loader instead of hitting the network.

    Raises CorpusLoadError(month) if a Monthly_Config fails to load, identifying
    which month failed (Requirement 1.4).
    """
    accumulator: list[str] = []

    for month in list_available_months():
        if len(accumulator) >= max_rows:
            break

        try:
            records = month_loader(month)
        except Exception as exc:
            raise CorpusLoadError(month) from exc

        for record in records:
            combined = combine_title_description(record["title"], record["description"])
            if combined is not None:
                accumulator.append(combined)

    return accumulator[:max_rows]
