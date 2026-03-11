import os
from pathlib import Path

data_path = Path("e:/code/rag_test/WikiEval/data")
files = list(data_path.glob("*.parquet"))

print("Parquet files:", [f.name for f in files])

if files:
    import pandas as pd
    df = pd.read_parquet(files[0])
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst row preview:")
    for col in df.columns[:3]:
        val = df[col].iloc[0]
        if isinstance(val, str):
            print(f"  {col}: {val[:100]}...")
        elif isinstance(val, list):
            print(f"  {col}: list with {len(val)} items")
        else:
            print(f"  {col}: {val}")
