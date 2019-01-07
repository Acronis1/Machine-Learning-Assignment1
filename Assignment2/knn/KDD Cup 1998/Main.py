import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import time

pd.set_option('display.max_columns', 50)

trainData = pd.read_csv("../data/KDD Cup/cup98ID.shuf.5000.train.csv", index_col="CONTROLN")
testData = pd.read_csv("../data/KDD Cup/cup98ID.shuf.5000.test.csv", index_col="CONTROLN")
target = trainData["TARGET_D"]

testIndex = testData

trainData.replace([" ", "", None], [np.nan, np.nan, np.nan], inplace=True)
testData.replace([" ", "", None], [np.nan, np.nan, np.nan], inplace=True)
# trainData.drop(columns="ZIP", inplace=True)
# testData.drop(columns="ZIP", inplace=True)
print(str(trainData.shape) + ", " + str(testData.shape))
trainData.drop(columns="TARGET_D", inplace=True)

data = pd.get_dummies(trainData.append(testData))
print(len(trainData))
cols = [i for i in data.columns]
for col in cols:
    data[col] = pd.to_numeric(data[col])

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
data = imp.fit_transform(data)

#data.fillna(method="ffill", inplace=True)
print(pd.isnull(data).any(1).nonzero()[0])
trainData = data[:len(trainData), :]
testData = data[len(trainData):, :]
print(data)
print(testData)

# data = data[data.columns[data.isnull().mean() < 0.8]]


#testData = pd.get_dummies(data)
# testData = testData[testData.columns[testData.isnull().mean() < 0.8]]
#testData.fillna(method="ffill", inplace=True)

# data.dropna(axis=1, inplace=True)
# testData.dropna(axis=1, inplace=True)
# print(data.head())
# print(testData.head())
#data.fillna("speck", inplace=True)


#print([i for i, j in zip(trainData.index, testData.index) if i != j])
print(str(trainData.shape) + ", " + str(testData.shape))
#print(trainData.isnull().any().any())
#print(testData.isnull().any().any())

selector = SelectKBest(f_regression, k=20).fit(trainData, target)
selectedTrainData = selector.transform(trainData)
selectedTestData = selector.transform(testData)
print("selected: "+str(selectedTrainData))

# scaling
# scaler = preprocessing.StandardScaler().fit(selectedTrainData + selectedTestData)
# selectedTrainData = scaler.transform(selectedTrainData)
# selectedTestData = scaler.transform(selectedTestData)

#print(trainData.isnull().any().any())
#print(testData.isnull().any().any())

# grid search
k_range = list(range(10, 25))
weights = ['uniform', 'distance']
algorithms = ['ball_tree', 'kd_tree', 'brute']
leafSize = list(range(1, 15))
p = list(range(1, 4))
param_grid = dict(n_neighbors=k_range, weights=weights, algorithm=algorithms, leaf_size=leafSize, p=p)
# print(target.head())
# print(data.head())

knn = KNeighborsRegressor()
grid = GridSearchCV(knn, param_grid, cv=5, scoring=['neg_mean_squared_error', 'r2'], refit='neg_mean_squared_error', n_jobs=-1, verbose=1)

grid.fit(selectedTrainData, target)

# print(grid.cv_results_)
print(grid.best_params_)
print(grid.best_score_)

bestKnn = grid.best_estimator_

prediction = bestKnn.predict(selectedTestData)
print(prediction)

prediction = pd.Series(prediction, index=testIndex.index)

counter = len([name for name in os.listdir("results") if os.path.isfile(os.path.join("results", name))])
prediction.to_csv("results/result"+str(counter)+".csv", header=['TARGET_D'])

counter = len([name for name in os.listdir("./") if os.path.isfile(name)])
results = pd.DataFrame(grid.cv_results_)
results.to_csv("results"+str(counter), columns=["mean_fit_time", "param_algorithm", "param_leaf_size", "param_n_neighbors", "param_p", "param_weights", "mean_test_neg_mean_squared_error", "rank_test_neg_mean_squared_error", "mean_test_r2", "rank_test_r2"],
               header=["mean_fit_time", "algorithm", "leaf_size", "neighbors", "param_p", "weights",  "mean_test_nmse", "rank_test_nmse", "mean_test_r2", "rank_test_r2"])
