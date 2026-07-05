from pathlib import Path

import pandas as pd

from .preprocessing import clean_text


def _validate_dataframe(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"Dataset is empty: {source}")

    missing_columns = {"sentence", "sentiment"} - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {source}: {', '.join(sorted(missing_columns))}"
        )

    return df


def prepare_dataset(train_path: str, test_path: str) -> pd.DataFrame:
    train_path_obj = Path(train_path)
    test_path_obj = Path(test_path)

    if not train_path_obj.is_file():
        raise FileNotFoundError(f"Train dataset not found: {train_path}")
    if not test_path_obj.is_file():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")

    train_df = pd.read_csv(train_path_obj)
    test_df = pd.read_csv(test_path_obj)

    train_df = _validate_dataframe(train_df, train_path)
    test_df = _validate_dataframe(test_df, test_path)

    train_df["dataset_type"] = "train"
    test_df["dataset_type"] = "test"

    merged_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    merged_df["clean_sentence"] = merged_df["sentence"].astype(str).map(clean_text)

    return merged_df
