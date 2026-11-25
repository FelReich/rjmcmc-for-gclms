"""Center and scale each Sachs protein column using its sample SD."""

from pathlib import Path

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[4]
SOURCE = CODE_ROOT / "data" / "cd3cd28_u0126.csv"
OUTPUT = Path(__file__).resolve().with_name("preprocessed_protein_data.csv")


def main():
    data = pd.read_csv(SOURCE)
    if len(data) < 2 or data.shape[1] < 2 or data.columns.has_duplicates:
        raise ValueError("Expected at least two observations and unique protein columns")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in data.dtypes):
        raise ValueError("All protein columns must be numeric")

    values = data.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Protein data contain missing or non-finite values")

    means = data.mean(axis=0)
    sample_sds = data.std(axis=0, ddof=1)
    if not np.isfinite(sample_sds).all() or (sample_sds <= 0).any():
        raise ValueError("Every protein must have a positive, finite sample SD")

    standardized = (data - means) / sample_sds
    standardized.to_csv(OUTPUT, index=False, float_format="%.17g")
    print(f"Saved {len(standardized)} observations and {standardized.shape[1]} proteins to {OUTPUT}")


if __name__ == "__main__":
    main()
