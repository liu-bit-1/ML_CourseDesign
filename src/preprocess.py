"""Preprocessing pipeline for the machine learning course design project.

This module only prepares data for later modeling. It does not train, evaluate,
or select any model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "preprocessed"

TARGET_COLUMN = "risk_label"
SESSION_ID_COLUMN = "session_id"
DATE_COLUMN = "op_date"
MISSING_CATEGORY = "__MISSING__"
UNKNOWN_CATEGORY = "UNKNOWN"

LOCATION_LEVEL_COLUMNS = ["first_lvl", "sec_lvl", "third_lvl"]
FREQUENCY_COLUMNS = [
    "user_name",
    "ip",
    "device_model",
    "browser_type",
    "os_type",
    "bus_system_code",
]
USER_AGGREGATE_COLUMNS = [
    "ip",
    "device_model",
    "browser_type",
    "bus_system_code",
]
PROTECTED_ENGINEERED_COLUMNS = {
    "session_prefix",
    "session_user_part_matches_user_name",
    "session_contains_access",
    "session_contains_login",
    "session_contains_sso",
    "user_auth_count",
    "user_ip_nunique",
    "user_device_model_nunique",
    "user_browser_type_nunique",
    "user_bus_system_code_nunique",
    "ip_auth_count",
    "ip_user_nunique",
}


@dataclass
class PreprocessedData:
    """Container for preprocessed train/test data and submission identifiers."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    train_session_id: pd.Series
    test_session_id: pd.Series


def find_data_file(candidates: list[str], data_dir: Path = DATA_DIR) -> Path:
    """Find the first existing file from a list of candidate names."""
    for filename in candidates:
        file_path = data_dir / filename
        if file_path.exists():
            return file_path
    raise FileNotFoundError(f"Could not find any of these files in {data_dir}: {candidates}")


def read_csv_auto_sep(file_path: Path) -> pd.DataFrame:
    """Read a CSV file that may be tab-separated or comma-separated."""
    df = pd.read_csv(file_path, sep="\t")

    # Data files are expected to be tab-separated; if parsing produces one
    # column, fall back to comma-separated reading for files such as submit CSVs.
    if df.shape[1] == 1:
        comma_df = pd.read_csv(file_path, sep=",")
        if comma_df.shape[1] > df.shape[1]:
            df = comma_df

    df.columns = df.columns.astype(str).str.strip()
    return df


def load_raw_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and submit example files from the data directory."""
    train_path = find_data_file(["train_dataset(1).csv", "train_dataset.csv"], data_dir)
    test_path = find_data_file(["test_dataset(1).csv", "test_dataset.csv"], data_dir)
    submit_path = find_data_file(["submit_example(1).csv", "submit_example.csv"], data_dir)

    train = read_csv_auto_sep(train_path)
    test = read_csv_auto_sep(test_path)
    submit = read_csv_auto_sep(submit_path)
    return train, test, submit


def validate_raw_data(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Validate columns required by the preprocessing pipeline."""
    required_train_columns = {SESSION_ID_COLUMN, DATE_COLUMN, TARGET_COLUMN}
    required_test_columns = {SESSION_ID_COLUMN, DATE_COLUMN}

    missing_train_columns = required_train_columns - set(train.columns)
    missing_test_columns = required_test_columns - set(test.columns)

    if missing_train_columns:
        raise ValueError(f"Training data is missing columns: {sorted(missing_train_columns)}")
    if missing_test_columns:
        raise ValueError(f"Test data is missing columns: {sorted(missing_test_columns)}")
    if TARGET_COLUMN in test.columns:
        raise ValueError(f"Test data should not contain target column: {TARGET_COLUMN}")


def add_time_features(df: pd.DataFrame, date_column: str = DATE_COLUMN) -> pd.DataFrame:
    """Extract basic datetime features from op_date."""
    processed = df.copy()
    op_datetime = pd.to_datetime(processed[date_column], errors="coerce")

    processed["op_year"] = op_datetime.dt.year
    processed["op_month"] = op_datetime.dt.month
    processed["op_day"] = op_datetime.dt.day
    processed["op_hour"] = op_datetime.dt.hour
    processed["op_weekday"] = op_datetime.dt.weekday

    return processed


def parse_location_value(value) -> dict[str, str]:
    """Parse the JSON-like location field into three location levels."""
    if pd.isna(value):
        return {column: UNKNOWN_CATEGORY for column in LOCATION_LEVEL_COLUMNS}

    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {column: UNKNOWN_CATEGORY for column in LOCATION_LEVEL_COLUMNS}

    if not isinstance(parsed, dict):
        return {column: UNKNOWN_CATEGORY for column in LOCATION_LEVEL_COLUMNS}

    return {
        column: str(parsed.get(column) or UNKNOWN_CATEGORY)
        for column in LOCATION_LEVEL_COLUMNS
    }


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract first_lvl, sec_lvl, and third_lvl from location."""
    processed = df.copy()
    if "location" not in processed.columns:
        for column in LOCATION_LEVEL_COLUMNS:
            processed[column] = UNKNOWN_CATEGORY
        return processed

    location_parts = processed["location"].apply(parse_location_value).apply(pd.Series)
    for column in LOCATION_LEVEL_COLUMNS:
        processed[column] = location_parts[column].fillna(UNKNOWN_CATEGORY).astype(str)

    return processed


def split_ip_prefixes(value) -> tuple[str, str, str]:
    """Extract first, first two, and first three IP segments."""
    if pd.isna(value):
        return UNKNOWN_CATEGORY, UNKNOWN_CATEGORY, UNKNOWN_CATEGORY

    parts = str(value).split(".")
    if len(parts) < 1 or not parts[0]:
        return UNKNOWN_CATEGORY, UNKNOWN_CATEGORY, UNKNOWN_CATEGORY

    ip_prefix_1 = parts[0]
    ip_prefix_2 = ".".join(parts[:2]) if len(parts) >= 2 else UNKNOWN_CATEGORY
    ip_prefix_3 = ".".join(parts[:3]) if len(parts) >= 3 else UNKNOWN_CATEGORY
    return ip_prefix_1, ip_prefix_2, ip_prefix_3


def add_ip_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract risk-control prefix features from IP."""
    processed = df.copy()
    if "ip" not in processed.columns:
        processed["ip_prefix_1"] = UNKNOWN_CATEGORY
        processed["ip_prefix_2"] = UNKNOWN_CATEGORY
        processed["ip_prefix_3"] = UNKNOWN_CATEGORY
        return processed

    ip_prefixes = processed["ip"].apply(split_ip_prefixes)
    processed["ip_prefix_1"] = ip_prefixes.apply(lambda value: value[0])
    processed["ip_prefix_2"] = ip_prefixes.apply(lambda value: value[1])
    processed["ip_prefix_3"] = ip_prefixes.apply(lambda value: value[2])
    return processed


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract structured features from session_id."""
    processed = df.copy()
    if SESSION_ID_COLUMN not in processed.columns:
        processed["session_prefix"] = UNKNOWN_CATEGORY
        processed["session_user_part_matches_user_name"] = 0
        processed["session_contains_access"] = 0
        processed["session_contains_login"] = 0
        processed["session_contains_sso"] = 0
        return processed

    session_text = processed[SESSION_ID_COLUMN].fillna("").astype(str)
    session_parts = session_text.str.split(":", expand=True)

    processed["session_prefix"] = session_parts[0].fillna(UNKNOWN_CATEGORY).replace("", UNKNOWN_CATEGORY)
    session_user_part = session_parts[1].fillna("").astype(str) if session_parts.shape[1] > 1 else ""

    if "user_name" in processed.columns:
        user_name = processed["user_name"].fillna("").astype(str)
        processed["session_user_part_matches_user_name"] = (session_user_part == user_name).astype(int)
    else:
        processed["session_user_part_matches_user_name"] = 0

    lower_session = session_text.str.lower()
    processed["session_contains_access"] = lower_session.str.contains("access", regex=False).astype(int)
    processed["session_contains_login"] = lower_session.str.contains("login", regex=False).astype(int)
    processed["session_contains_sso"] = lower_session.str.contains("sso", regex=False).astype(int)

    return processed


def add_basic_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add row-level risk-control features before train-based statistics."""
    processed = add_location_features(df)
    processed = add_ip_features(processed)
    processed = add_session_features(processed)
    return processed


def normalized_key(series: pd.Series) -> pd.Series:
    """Normalize categorical keys before count and aggregate mapping."""
    return series.fillna(MISSING_CATEGORY).astype(str)


def add_frequency_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add frequency features computed from training data only."""
    train = train.copy()
    test = test.copy()

    for column in FREQUENCY_COLUMNS:
        if column not in train.columns or column not in test.columns:
            continue

        counts = normalized_key(train[column]).value_counts()
        feature_name = f"{column}_count"
        train[feature_name] = normalized_key(train[column]).map(counts).fillna(0).astype(int)
        test[feature_name] = normalized_key(test[column]).map(counts).fillna(0).astype(int)

    return train, test


def map_aggregate_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    key_column: str,
    aggregate_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map train-only aggregate statistics back to train and test."""
    key_train = normalized_key(train[key_column])
    key_test = normalized_key(test[key_column])

    for feature_name in aggregate_df.columns:
        train[feature_name] = key_train.map(aggregate_df[feature_name]).fillna(0).astype(int)
        test[feature_name] = key_test.map(aggregate_df[feature_name]).fillna(0).astype(int)

    return train, test


def add_user_aggregate_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add user-level behavior aggregates computed from training data only."""
    if "user_name" not in train.columns or "user_name" not in test.columns:
        return train, test

    train = train.copy()
    test = test.copy()
    working = train.copy()
    working["_user_key"] = normalized_key(working["user_name"])

    aggregate_spec = {
        "user_auth_count": (SESSION_ID_COLUMN, "count"),
    }
    for column in USER_AGGREGATE_COLUMNS:
        if column in working.columns:
            feature_name = f"user_{column}_nunique"
            aggregate_spec[feature_name] = (
                column,
                lambda value: normalized_key(value).nunique(dropna=False),
            )

    user_aggregates = working.groupby("_user_key").agg(**aggregate_spec)
    return map_aggregate_features(train, test, "user_name", user_aggregates)


def add_ip_aggregate_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add IP-level aggregates computed from training data only."""
    if "ip" not in train.columns or "ip" not in test.columns:
        return train, test

    train = train.copy()
    test = test.copy()
    working = train.copy()
    working["_ip_key"] = normalized_key(working["ip"])

    aggregate_spec = {
        "ip_auth_count": (SESSION_ID_COLUMN, "count"),
    }
    if "user_name" in working.columns:
        aggregate_spec["ip_user_nunique"] = (
            "user_name",
            lambda value: normalized_key(value).nunique(dropna=False),
        )

    ip_aggregates = working.groupby("_ip_key").agg(**aggregate_spec)
    return map_aggregate_features(train, test, "ip", ip_aggregates)


def add_train_based_risk_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add count and aggregate features using train-only statistics."""
    train, test = add_frequency_features(train, test)
    train, test = add_user_aggregate_features(train, test)
    train, test = add_ip_aggregate_features(train, test)
    return train, test


def split_features_and_ids(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Separate model features, target values, and session IDs."""
    y_train = train[TARGET_COLUMN].copy()
    train_session_id = train[SESSION_ID_COLUMN].copy()
    test_session_id = test[SESSION_ID_COLUMN].copy()

    drop_columns = [TARGET_COLUMN, SESSION_ID_COLUMN, DATE_COLUMN]
    train_features = train.drop(columns=[col for col in drop_columns if col in train.columns]).copy()
    test_features = test.drop(columns=[col for col in drop_columns if col in test.columns]).copy()

    return train_features, test_features, y_train, train_session_id, test_session_id


def fill_missing_values(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill missing values using one consistent rule for train and test."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [col for col in X_train.columns if col not in numeric_columns]

    for column in numeric_columns:
        median_value = X_train[column].median()
        if pd.isna(median_value):
            median_value = -1
        X_train[column] = X_train[column].fillna(median_value)
        X_test[column] = X_test[column].fillna(median_value)

    for column in categorical_columns:
        X_train[column] = X_train[column].fillna(MISSING_CATEGORY).astype(str)
        X_test[column] = X_test[column].fillna(MISSING_CATEGORY).astype(str)

    return X_train, X_test


def drop_constant_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    protected_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop columns that have only one value across train and test."""
    protected_columns = protected_columns or set()
    combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    constant_columns = [
        column
        for column in combined.columns[combined.nunique(dropna=False) <= 1].tolist()
        if column not in protected_columns
    ]

    if constant_columns:
        X_train = X_train.drop(columns=constant_columns)
        X_test = X_test.drop(columns=constant_columns)

    return X_train, X_test


def encode_categorical_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode categorical variables with aligned train/test columns."""
    train_size = len(X_train)
    combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    categorical_columns = combined.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    combined_encoded = pd.get_dummies(
        combined,
        columns=categorical_columns,
        dummy_na=False,
        dtype=np.int8,
    )

    X_train_encoded = combined_encoded.iloc[:train_size].reset_index(drop=True)
    X_test_encoded = combined_encoded.iloc[train_size:].reset_index(drop=True)

    return X_train_encoded, X_test_encoded


def preprocess_data(train: pd.DataFrame, test: pd.DataFrame) -> PreprocessedData:
    """Apply the full preprocessing pipeline to train and test data."""
    validate_raw_data(train, test)

    train_with_time = add_time_features(train)
    test_with_time = add_time_features(test)
    train_with_features = add_basic_risk_features(train_with_time)
    test_with_features = add_basic_risk_features(test_with_time)
    train_with_features, test_with_features = add_train_based_risk_features(
        train_with_features,
        test_with_features,
    )

    X_train, X_test, y_train, train_session_id, test_session_id = split_features_and_ids(
        train_with_features,
        test_with_features,
    )
    X_train, X_test = fill_missing_values(X_train, X_test)
    X_train, X_test = drop_constant_columns(
        X_train,
        X_test,
        protected_columns=PROTECTED_ENGINEERED_COLUMNS,
    )
    X_train, X_test = encode_categorical_features(X_train, X_test)

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train and test feature columns are not aligned after preprocessing.")

    return PreprocessedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train.reset_index(drop=True),
        train_session_id=train_session_id.reset_index(drop=True),
        test_session_id=test_session_id.reset_index(drop=True),
    )


def save_preprocessed_data(
    preprocessed: PreprocessedData,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Save preprocessed outputs for later model training and submission."""
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessed.X_train.to_csv(output_dir / "X_train.csv", index=False)
    preprocessed.X_test.to_csv(output_dir / "X_test.csv", index=False)
    preprocessed.y_train.to_frame(TARGET_COLUMN).to_csv(output_dir / "y_train.csv", index=False)
    preprocessed.test_session_id.to_frame(SESSION_ID_COLUMN).to_csv(
        output_dir / "test_session_id.csv",
        index=False,
    )


def main() -> None:
    """Run preprocessing and save reusable outputs without training any model."""
    train, test, _submit = load_raw_data()
    preprocessed = preprocess_data(train, test)
    save_preprocessed_data(preprocessed)

    print("Preprocessing completed. No model training was started.")
    print(f"X_train shape: {preprocessed.X_train.shape}")
    print(f"X_test shape: {preprocessed.X_test.shape}")
    print(f"y_train shape: {preprocessed.y_train.shape}")
    print(f"Saved files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
