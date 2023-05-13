import pandas as pd

import numpy as np


df = pd.read_csv("./imdb_movies.csv/imdb_movies.csv")

print(df.head())

df = df.drop(["overview", "crew","status","orig_lang", "orig_title"], axis=1)

media = df['score'].mean()

print("La media del score es:", media)

mediana = df['score'].median()

print('La mediana del score es:', mediana)

moda = df['score'].mode()

print('La moda del score es:', moda)

rango = np.ptp(df['budget_x'])

print("El rango del budget es:", rango)

devstan = np.std(df['score'])

print("La desviación estándar del score es:", devstan)
