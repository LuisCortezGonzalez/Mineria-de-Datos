import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./imdb_movies.csv/imdb_movies.csv")

print(df.head())

df = df.drop(["overview", "crew","status","orig_lang", "orig_title"], axis=1)

df_genero = df.groupby('genre')['names'].count().reset_index()

fig, ax = plt.subplots(figsize=(8,6))
ax.bar(df_genero['genre'], df_genero['names'], color='b')
ax.set_xlabel('Género')
ax.set_ylabel('Número de películas')
ax.set_title('Número de películas por género')
plt.show()
