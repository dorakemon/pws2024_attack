import pandas as pd

pd.read_csv("./E16.csv", header=None).T.to_csv("./E16_T.csv", header=None, index=None)
