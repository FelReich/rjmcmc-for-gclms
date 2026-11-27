"""Standardize the Sachs protein data after removing Raf."""

from pathlib import Path

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[4]
SOURCE = CODE_ROOT / "data" / "cd3cd28_u0126.csv"
OUTPUT = Path(__file__).resolve().with_name("preprocessed_remove_protein.csv")


def main() -> None:
    data = pd.read_csv(SOURCE)
    if data.columns.has_duplicates or "Raf" not in data.columns:
        raise ValueError("Expected unique protein columns including Raf")
    data = data.drop(columns="Raf")
    if len(data) < 2 or data.shape[1] != 10:
        raise ValueError("Expected at least two observations of ten proteins")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in data.dtypes):
        raise ValueError("All retained protein columns must be numeric")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Retained protein data contain missing or non-finite values")

    sample_sds = data.std(axis=0, ddof=1)
    if not np.isfinite(sample_sds).all() or (sample_sds <= 0).any():
        raise ValueError("Every retained protein must have a positive, finite sample SD")
    standardized = (data - data.mean(axis=0)) / sample_sds
    standardized.to_csv(OUTPUT, index=False, float_format="%.17g")
    print(f"Saved {len(standardized)} observations of ten proteins to {OUTPUT}")


if __name__ == "__main__":
    main()
