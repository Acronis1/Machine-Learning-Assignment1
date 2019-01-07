import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
import time

data = pd.read_csv("../data/breast cancer/wpbc.data", index_col=0, header=None)
data.replace(["R", "N", "?"], [0, 1, np.nan], inplace=True)

target = data[2]
data.drop(columns=[2], inplace=True)

indexData = data.index

# data.dropna(inplace=True)
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data = imp.fit_transform(data)



print(data)
print(target)

# scaling
# scaler = preprocessing.StandardScaler().fit(data)
# data = scaler.transform(data)

print(data)

trainData, testData = train_test_split(data, test_size=0.2, shuffle=False)
trainTarget, testTarget = train_test_split(target, test_size=0.2, shuffle=False)

k_range = list(range(1, 20))
weights = ['uniform', 'distance']
algorithms = ['ball_tree', 'kd_tree', 'brute']
leafSize = list(range(10, 20))
p = list(range(1, 4))
param_grid = dict(n_neighbors=k_range, weights=weights, algorithm=algorithms, leaf_size=leafSize, p=p)


knn = KNeighborsRegressor()
grid = GridSearchCV(knn, param_grid, cv=5, scoring=['neg_mean_squared_error', 'r2'], refit='neg_mean_squared_error', n_jobs=-1, verbose=1)

grid.fit(trainData, trainTarget)

# print(grid.cv_results_)
print(grid.best_params_)
print(grid.best_score_)

bestKnn = grid.best_estimator_

prediction = bestKnn.predict(testData)

print("mean_squared_error: "+str(mean_squared_error(testTarget, prediction)))
print("r2: "+str(bestKnn.score(testData, testTarget)))

counter = len([name for name in os.listdir("./") if os.path.isfile(name)])
results = pd.DataFrame(grid.cv_results_)
results.to_csv("results"+str(counter), columns=["mean_fit_time", "param_algorithm", "param_leaf_size", "param_n_neighbors", "param_p", "param_weights", "mean_test_neg_mean_squared_error", "rank_test_neg_mean_squared_error", "mean_test_r2", "rank_test_r2"],
               header=["mean_fit_time", "algorithm", "leaf_size", "neighbors", "param_p", "weights",  "mean_test_nmse", "rank_test_nmse", "mean_test_r2", "rank_test_r2"])
