from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


df = pd.read_csv("./imdb_movies.csv/imdb_movies.csv")

print(df.head())

df = df.drop(["overview", "crew","status","orig_lang", "orig_title"], axis=1)


X = df[['score', 'budget_x', 'revenue']]
y = df['names']


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = model.score(X_test, y_test)
print('Precisión:', accuracy)
