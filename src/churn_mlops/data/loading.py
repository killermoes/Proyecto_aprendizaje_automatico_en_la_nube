import pandas as pd
from sklearn.model_selection import train_test_split

from configs import DATA_PATH, RANDOM_STATE, TEST_SIZE


def load_and_split_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Prepared dataset was not found: {DATA_PATH}")

    dataframe = pd.read_csv(DATA_PATH)
    required_columns = {"customerID", "Churn"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    features = dataframe.drop(columns=["customerID", "Churn"])
    integer_columns = features.select_dtypes(include=["integer"]).columns
    features[integer_columns] = features[integer_columns].astype("float64")

    target = dataframe["Churn"]
    if target.isna().any():
        raise ValueError("The target column 'Churn' contains missing values.")
    if set(target.unique()) != {"No", "Yes"}:
        raise ValueError("The target column 'Churn' must contain 'No' and 'Yes'.")
    target = target.map({"No": 0, "Yes": 1}).astype("int64")

    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
