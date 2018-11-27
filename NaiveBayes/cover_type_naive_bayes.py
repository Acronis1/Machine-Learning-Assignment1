from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import time


# sm = SMOTE()
df = pd.read_csv("../dataSets/covtype.data", header=None)
df = df.sample(frac=1)
classes = df.loc[:, 54]
print(df.loc[:, 54].value_counts())
# data, classes = sm.fit_resample(df, classes)
continous = df.loc[:, 0:9]
binary = df.loc[:, 10:53]
print(classes.head())
print(continous.head())
print(binary.head())


classesTrain, classesTest = train_test_split(classes, test_size=0.2)
continousTrain, continousTest = train_test_split(continous, test_size=0.2)
binaryTrain, binaryTest = train_test_split(binary, test_size=0.2)

# print(df[54].value_counts().index)
occurrences = df[54].value_counts()
maxOccurrence = occurrences.iloc[0]
weights = []

# for i in range(0, len(occurrences)):
#     weights.append(maxOccurrence/occurrences.iloc[i])
# weights = classes.replace(to_replace=df[54].value_counts().index, value=weights)

# weightsTrain, weightsTest = train_test_split(weights, test_size=0.2)
# print(classes.head())
# print(continous.head())
# print(binary.head())
start_time = time.time()
modelBernoulli = BernoulliNB(alpha=1)
modelBernoulli.fit(binaryTrain, classesTrain)

modelGausian = GaussianNB()
modelGausian.fit(continousTrain, classesTrain)

# print(modelBernoulli.predict_proba(binaryTrain))
# print(modelGausian.predict_proba(continousTrain))

dfTrain = np.hstack((modelGausian.predict_proba(continousTrain), modelBernoulli.predict_proba(binaryTrain)))

modelFinal = MultinomialNB()
modelFinal.fit(dfTrain, classesTrain)
print("--- %s seconds ---" % (time.time() - start_time))

# Predict Output
dfTest = np.hstack((modelGausian.predict_proba(continousTest), modelBernoulli.predict_proba(binaryTest)))

predicted = modelFinal.predict(dfTest)

predicted = pd.Series(predicted)

# print(df)

print('Training Score: {0}'.format(modelFinal.score(dfTrain,classesTrain)))
print('Testing Score: {0}'.format(modelFinal.score(dfTest, classesTest)))

target_names = df[54].unique
print(classification_report(classesTest, predicted))
print(confusion_matrix(classesTest, predicted))

#counter = len([name for name in os.listdir("results") if os.path.isfile(os.path.join("results", name))])
#predicted.to_csv("results/result"+str(counter)+".csv", header=['class'])
