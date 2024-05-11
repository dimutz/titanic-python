"""
	Train Titanic Dataset
"""
import pandas as pd

# TASK 1
titanic = pd.read_csv('data/train.csv')
(nr_rows, nr_cols) = titanic.shape
print(f"Number of lines and columns: {nr_rows}, {nr_cols}")
print(titanic.dtypes)
columns = titanic.columns.tolist()
for col in columns:
    full_rows = titanic[col].dropna().count()
    print(f"Column '{col}' missing {nr_rows - full_rows} values")
duped = titanic.loc[titanic.index.duplicated(), :]
print(f"Duplicate lines? {duped.shape[0] != 0}")
