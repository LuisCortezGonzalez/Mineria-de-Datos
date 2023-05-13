import pandas as pd

from sklearn.linear_model import LinearRegression

df = pd.read_csv("./imdb_movies.csv/imdb_movies.csv")

df = df.drop(["overview", "crew","status","orig_lang", "orig_title"], axis=1)

df['revenue'] = df['revenue'].astype(int)

print(df.head())

X = df[['budget_x', 'revenue']]
y = df['score']

lr = LinearRegression()
lr.fit(X, y)
print(lr.coef_)
prediction = lr.predict([[50000000, 10000000]])
print(prediction)
