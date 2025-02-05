import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("utils")


def validate_and_correct_data(df, target_column):
    """
    Cleans and prepares the dataset for ML training.
    - Converts date columns to useful features.
    - Converts non-numeric columns to numeric if possible.
    - Handles missing values.
    - Removes duplicates.
    """
    df = df.copy()

    # 1️⃣ **Check for missing values and fill/remove them**
    if df.isnull().sum().sum() > 0:
        logger.warning("⚠️ Missing values detected! Handling them...")
        df = df.ffill().bfill()

    # 2️⃣ **Convert Date column (if it exists) to numeric features**
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            logger.debug(f"🕒 Converting datetime column: {col}")
            df[f"{col}_Year"] = df[col].dt.year
            df[f"{col}_Month"] = df[col].dt.month
            df[f"{col}_Day"] = df[col].dt.day
            df[f"{col}_DayOfWeek"] = df[col].dt.dayofweek
            df.drop(columns=[col], inplace=True)

    # 3️⃣ **Convert object (string) columns to numeric if possible**
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert numeric strings
            logger.debug(f"🔢 Converted {col} to numeric")
        except:
            logger.warning(
                f"⚠️ Non-numeric column '{col}' detected. Consider encoding it."
            )

    # 4️⃣ **Ensure no missing values after conversion**
    df = df.fillna(0)  # Replace any remaining NaN values with 0

    # 5️⃣ **Remove duplicate rows**
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    if df.shape[0] < initial_rows:
        logger.info(f"🧹 Removed {initial_rows - df.shape[0]} duplicate rows.")

    # 6️⃣ **Ensure target column exists**
    if target_column not in df.columns:
        raise ValueError(
            f"❌ Target column '{target_column}' is missing from the dataset!"
        )

    logger.info("✅ Data validation & correction complete!")
    return df


def create_dask_dataframe(n_samples=10000):
    """Generates a realistic dataset representing social connections or interactions."""
    np.random.seed(42)

    users = np.random.randint(1000, 1100, size=n_samples)  # User IDs (1000-1099)
    friends = np.random.randint(1000, 1100, size=n_samples)  # Friend connections
    interaction_strength = (
        np.random.rand(n_samples) * 10
    )  # Random weights for interaction strength

    df = pd.DataFrame(
        {"User": users, "Friend": friends, "Strength": interaction_strength}
    )
    return df


def compute_class_weights(y):
    """Compute balanced class weights"""
    class_counts = y.value_counts()
    total = len(y)
    weights = {
        i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)
    }
    return weights


def make_unique_name(name, used_names):
    # Clean the name first
    clean_name = name.replace(":", "_").replace("-", "_")
    clean_name = "".join(c for c in clean_name if c.isalnum() or c == "_")
    base_name = clean_name[:8]  # Leave room for numbers

    if base_name not in used_names:
        used_names.add(base_name)
        return base_name

    # If name is already used, add a number
    counter = 1
    while True:
        numbered_name = f"{base_name[:8-len(str(counter))]}{counter}"
        if numbered_name not in used_names:
            used_names.add(numbered_name)
            return numbered_name
        counter += 1
