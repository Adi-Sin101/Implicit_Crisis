"""Back up split CSVs and remove only the raw severity column."""

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "final_datasets" / "splits"
FILES = ("train.csv", "validation.csv", "test.csv")
DROP_COLUMN = "severity"
EXPECTED_RETAINED_COLUMNS = [
    "row_index",
    "url",
    "content",
    "label",
    "class_name",
    "split",
]


def process_file(filename: str) -> None:
    path = SPLITS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing split file: {path}")

    before = pd.read_csv(path)
    before_columns = before.columns.tolist()
    before_rows = len(before)
    print(f"{filename} before: columns={before_columns}, rows={before_rows}")

    if DROP_COLUMN not in before.columns:
        raise ValueError(f"{filename} does not contain required column '{DROP_COLUMN}'")
    if [column for column in before_columns if column != DROP_COLUMN] != EXPECTED_RETAINED_COLUMNS:
        raise ValueError(f"Unexpected columns in {filename}: {before_columns}")

    backup_path = path.with_name(path.name + ".bak")
    before.to_csv(backup_path, index=False)

    after = before.drop(columns=[DROP_COLUMN])
    after.to_csv(path, index=False)

    result = pd.read_csv(path)
    if result.columns.tolist() != EXPECTED_RETAINED_COLUMNS:
        raise AssertionError(f"Column invariant failed for {filename}")
    if len(result) != before_rows:
        raise AssertionError(f"Row-count invariant failed for {filename}")
    if not result.equals(before[EXPECTED_RETAINED_COLUMNS]):
        raise AssertionError(f"Retained-value invariant failed for {filename}")

    print(f"{filename} after:  columns={result.columns.tolist()}, rows={len(result)}")
    print(f"Backup created: {backup_path.name}")


def main() -> None:
    for filename in FILES:
        process_file(filename)


if __name__ == "__main__":
    main()
