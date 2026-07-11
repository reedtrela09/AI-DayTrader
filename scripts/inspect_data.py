import pandas as pd
from pathlib import Path

project_folder = Path(__file__).parent.parent
data_folder = project_folder / "data"

df = pd.read_csv(data_folder / "spy.csv")

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10))