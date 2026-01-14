from pathlib import Path
import pandas as pd
from Dataset.dataset import Dataset
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAW_PATH = Path("Dataset/raw/ecommerce_fraud.csv")
CLEAN_PATH = Path("Dataset/processed/ecommerce_fraud_clean.csv")
REDUCED_PATH = Path("Dataset/processed/ecommerce_fraud_reduced.csv")

TARGET_COL = "is_fraud"

NUMERIC_FEATURES = [
    "account_age_days",
    "total_transactions_user",
    "avg_amount_user",
    "amount",
    "shipping_distance_km",
    "transaction_hour",
    "promo_used",
    "avs_match",
    "cvv_result",
    "three_ds_flag",
    "high_amount",
]

CATEGORICAL_FEATURES = [
    "country",
    "bin_country",
    "channel",
    "merchant_category",
]


def load_clean_dataset() -> pd.DataFrame:
    ds = Dataset(CLEAN_PATH)
    return ds.getDataset()


def load_reduced_dataset() -> pd.DataFrame:
    ds = Dataset(REDUCED_PATH)
    return ds.getDataset()


def build_clean_dataset_from_raw():
    ds = Dataset(RAW_PATH)
    df = ds.getDataset()

    print("Dataset raw caricato.")
    print("Shape raw:", df.shape)

    df["transaction_time"] = pd.to_datetime(df["transaction_time"])

    df["transaction_hour"] = df["transaction_time"].dt.hour

    if "high_amount" not in df.columns:
        df["high_amount"] = (df["amount"] > 300).astype(int)

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)

    print(f"Salvato dataset pulito in: {CLEAN_PATH}")
    print("Shape clean:", df.shape)

    if TARGET_COL in df.columns:
        print("Distribuzione is_fraud nel clean:")
        print(df[TARGET_COL].value_counts(normalize=True).to_string())


def create_reduced_dataset(n_nonfraud: int = 25000, random_state: int = 42):
    df = load_clean_dataset()

    df_fraud = df[df[TARGET_COL] == 1]
    df_nonfraud = df[df[TARGET_COL] == 0].sample(n=min(n_nonfraud, (df[TARGET_COL] == 0).sum()),random_state=random_state)

    df_reduced = pd.concat([df_fraud, df_nonfraud], axis=0).sample(frac=1.0,random_state=random_state)

    REDUCED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_reduced.to_csv(REDUCED_PATH, index=False)

    print(f"\nFrodi totali: {len(df_fraud)}")
    print(f"Non-frodi totali: {(df[TARGET_COL] == 0).sum()}")
    print(f"Shape ridotto: {df_reduced.shape}")
    print("Distribuzione is_fraud nel ridotto:")
    print(df_reduced[TARGET_COL].value_counts().to_string())
    print(f"Salvato dataset ridotto in: {REDUCED_PATH}")


def build_preprocessor() -> ColumnTransformer:
    transformers = []

    if NUMERIC_FEATURES:
        transformers.append(("num", StandardScaler(), NUMERIC_FEATURES))

    if CATEGORICAL_FEATURES:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    return preprocessor


def prepare_train_test_raw(test_size: float = 0.2, random_state: int = 42):
    df = load_reduced_dataset()

    cols_to_drop_from_X = ["user_id", "transaction_time", TARGET_COL]
    X_raw = df.drop(columns=cols_to_drop_from_X)
    y = df[TARGET_COL]

    print("\nShape X_raw:", X_raw.shape)
    print("Distribuzione y complessiva:")
    print(y.value_counts().to_string())

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("\nDistribuzione y_train (RAW):")
    print(y_train.value_counts().to_string())
    print("\nDistribuzione y_test (RAW):")
    print(y_test.value_counts().to_string())

    return X_train_raw, X_test_raw, y_train, y_test