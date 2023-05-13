import pandas as pd

df = pd.read_csv("./imdb_movies.csv/imdb_movies.csv")

print(df.head())

df = df.drop(["overview", "crew","status","orig_lang", "orig_title"], axis=1)

print(df)
