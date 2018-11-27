from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import time





df = pd.read_csv("../dataSets/mushroom_data.data")
df.fillna(df.mode().iloc[0], inplace=True)
# df.drop(columns=["veil-type", "gill-spacing"], inplace=True)
# df.dropna(axis=0, how="any")
# df = df.sample(frac=1)
# df['word'].value_counts()
print(df["class"].value_counts())

classes = df.get("class")
classes, stuff = train_test_split(classes, test_size=0.2, shuffle=False)
df.drop(columns=["class"], inplace=True)

# one hot encoding
#df = pd.get_dummies(df)

# label encoding
df = df.apply(LabelEncoder().fit_transform)

print(df.head())

train, test = train_test_split(df, test_size=0.2, shuffle=False)

start_time = time.time()

model = MultinomialNB(alpha=0.1)
model.fit(train, classes)

print("--- %s seconds ---" % (time.time() - start_time))
# Predict Output
predicted = model.predict(test)

predicted = pd.Series(predicted)

print(df)
print(df.corr())

print('Training Score: {0}'.format(model.score(train,classes)))
print('Testing Score: {0}'.format(model.score(test, stuff)))


print(classification_report(stuff, predicted))
print(confusion_matrix(stuff, predicted))

# counter = len([name for name in os.listdir("results") if os.path.isfile(os.path.join("results", name))])
# df.to_csv("results/result"+str(counter)+".csv", header=['class'])
