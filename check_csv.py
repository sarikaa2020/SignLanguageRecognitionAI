import pandas as pd

df = pd.read_csv("data/Hello/0.csv")
print(df.head())
print("Columns:", df.columns)
print("Dtypes:", df.dtypes)
