import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import time

pd.set_option('display.max_columns', 20)

# data
data = pd.read_csv("../data/AutoMPG/AutoMPG.shuf.train.csv", index_col="id")
data.replace("?", np.nan, inplace=True)
data["horsepower"] = pd.to_numeric(data["horsepower"])
# data.fillna(data.mean(), inplace=True)
data.dropna(inplace=True)

# test data
testData = pd.read_csv("../data/AutoMPG/AutoMPG.shuf.test.csv", index_col="id")
testData.replace("?", np.nan, inplace=True)
testData["horsepower"] = pd.to_numeric(testData["horsepower"])
# testData.fillna(testData.mean(), inplace=True)
testData.dropna(inplace=True)

# preprocessing label encoding
# le = LabelEncoder()
# labels = data["carName"].append(testData["carName"])
# le = le.fit(labels)
# data["carName"] = le.transform(data["carName"])
# testData["carName"] = le.transform(testData["carName"])

# preprocessing remove carName
data.drop(columns="carName", inplace=True)
testData.drop(columns="carName", inplace=True)

target = data["mpg"]
data.drop(columns=["mpg"], inplace=True)

indexData = testData

# scaling
scaler = preprocessing.StandardScaler().fit(data.append(testData))
data = scaler.transform(data)
print(data)
testData = scaler.transform(testData)

# data.dropna(inplace=True)



k_range = list(range(1, 25))
weights = ['uniform', 'distance']
algorithms = ['ball_tree', 'kd_tree', 'brute']
leafSize = list(range(1, 25))
p = list(range(1, 4))
param_grid = dict(n_neighbors=k_range, weights=weights, algorithm=algorithms, leaf_size=leafSize, p=p)
# print(target.head())
# print(data.head())

knn = KNeighborsRegressor()
grid = GridSearchCV(knn, param_grid, cv=5, scoring=['neg_mean_squared_error', 'r2'], refit='neg_mean_squared_error', n_jobs=-1, verbose=1)

# knn.fit(data, target)

grid.fit(data, target)

# print(grid.cv_results_)
print(grid.best_params_)
print(grid.best_score_)

bestKnn = grid.best_estimator_

prediction = bestKnn.predict(testData)
print(prediction)

prediction = pd.Series(prediction, index=indexData.index)

counter = len([name for name in os.listdir("results") if os.path.isfile(os.path.join("results", name))])
prediction.to_csv("results/result"+str(counter)+".csv", header=['mpg'])

counter = len([name for name in os.listdir("./") if os.path.isfile(name)])
results = pd.DataFrame(grid.cv_results_)
results.to_csv("results"+str(counter), columns=["mean_fit_time", "param_algorithm", "param_leaf_size", "param_n_neighbors", "param_p", "param_weights", "mean_test_neg_mean_squared_error", "rank_test_neg_mean_squared_error", "mean_test_r2", "rank_test_r2"],
               header=["mean_fit_time", "algorithm", "leaf_size", "neighbors", "param_p", "weights",  "mean_test_nmse", "rank_test_nmse", "mean_test_r2", "rank_test_r2"])

