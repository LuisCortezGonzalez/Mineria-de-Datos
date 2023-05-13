import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats


df = pd.read_csv("./imdb_movies.csv/imdb_movies.csv")


df_LIMP = pd.DataFrame.from_dict(df.loc[:,['names','date_x','score','budget_x','revenue']])
print(df_LIMP)

def get_prediction_interval(prediction, y_test, test_predictions, pi=.95):
    sum_errs = np.sum((y_test - test_predictions)**2)
    stdev = np.sqrt(1 / (len(y_test) - 2) * sum_errs)
    one_minus_pi = 1 - pi
    ppf_lookup = 1 - (one_minus_pi / 2)
    z_score = stats.norm.ppf(ppf_lookup)
    interval = z_score * stdev
    lower, upper = prediction - interval, prediction + interval
    return lower, prediction, upper


df_lr = pd.DataFrame({'budget_x': df_LIMP['budget_x'], 'revenue': df_LIMP['revenue']})
X = df_lr['budget_x'].values.reshape(-1,1)
Y= df_lr['revenue']

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

lower_vet = []
upper_vet = []

for i in Y_pred:
    lower, prediction, upper =  get_prediction_interval(i, df_lr['revenue'], Y_pred)
    lower_vet.append(lower)
    upper_vet.append(upper)

plt.fill_between(np.arange(0,len(df_lr['revenue']),1),upper_vet, lower_vet, color='b',label='IC = 0.95')
plt.xlabel('Budget')
plt.ylabel('Revenue')
plt.title('Puntuacion')
plt.show()
plt.close()