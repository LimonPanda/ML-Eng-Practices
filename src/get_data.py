"""Download and verify the original UCI CSV."""

import csv
import io
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "bank-full.csv"
DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
EXPECTED_COLUMNS = (
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "y",
)
EXPECTED_ROWS = 45_211


def main() -> None:
    print(f"Downloading {DATA_URL}")
    with urlopen(DATA_URL, timeout=60) as response:
        archive = io.BytesIO(response.read())
    with ZipFile(archive) as bundle:
        inner_archive = io.BytesIO(bundle.read("bank.zip"))
    with ZipFile(inner_archive) as bundle:
        data = bundle.read(DATA_PATH.name)

    reader = csv.reader(io.StringIO(data.decode("utf-8-sig")), delimiter=";")
    if tuple(next(reader, ())) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns in {DATA_PATH.name}")
    rows = 0
    for row in reader:
        if len(row) != len(EXPECTED_COLUMNS) or row[-1] not in {"yes", "no"}:
            raise ValueError(f"Invalid data in CSV line {reader.line_num}")
        rows += 1
    if rows != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {rows}")

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_bytes(data)
    print(f"Verified {rows:,} rows and saved {DATA_PATH.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
