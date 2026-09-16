"""Ingest + validate data.
Uses data/creditcard.csv (Kaggle) if present, else generates a synthetic
dataset with the same schema so the pipeline runs immediately."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

RAW = Path("data/creditcard.csv")
OUT = Path("data/processed.csv")
FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


def synthetic(n: int = 50_000) -> pd.DataFrame:
    X, y = make_classification(n_samples=n, n_features=30, n_informative=12,
                               weights=[0.983], flip_y=0.002, random_state=42)
    df = pd.DataFrame(X, columns=FEATURES)
    df["Time"] = np.random.default_rng(42).uniform(0, 172_800, n)
    df["Amount"] = np.abs(df["Amount"]) * 80
    df["Class"] = y
    return df


def validate(df: pd.DataFrame) -> None:
    missing = set(FEATURES + ["Class"]) - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    assert df[FEATURES].isnull().sum().sum() == 0, "Null values found"
    assert set(df["Class"].unique()) <= {0, 1}, "Label must be 0/1"
    assert (df["Amount"] >= 0).all(), "Negative amounts"


def main() -> None:
    df = pd.read_csv(RAW) if RAW.exists() else synthetic()
    print(f"Source: {'Kaggle' if RAW.exists() else 'synthetic'} | rows={len(df)} "
          f"| fraud rate={df['Class'].mean():.4%}")
    validate(df)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Validated data written to {OUT}")


if __name__ == "__main__":
    main()
