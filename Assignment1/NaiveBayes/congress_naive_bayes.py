from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import pandas as pd
import numpy as np
import os
import time




# pd.set_option('display.max_columns', 20)
df = pd.read_csv("../dataSets/congress_voting/CongressionalVotingID.shuf.train.csv", index_col="ID")

df = df.replace(["y", "n", "unknown"], ["1", "0", np.nan])
# df.fillna(df.mode().iloc[0], inplace=True)
df.dropna(axis=0, how="any", inplace=True)
# features.fillna(features.mode().iloc[0], inplace=True)

# dfRepub = df.loc[df['class'] == "republican"]
# dfDemo = df.loc[df['class'] == "democrat"]
# dfDemo = dfDemo.fillna(dfDemo.mode().iloc[0])
# dfRepub = dfRepub.fillna(dfRepub.mode().iloc[0])
# frames = [dfDemo, dfRepub]
# df = pd.concat(frames)
# df = df.sample(frac=1)

classes = df["class"]
df = df.drop(columns=["class"])

# prepare model
start_time = time.time()

model = MultinomialNB(alpha=0.001)
model.fit(df, classes)

print("--- %s seconds ---" % (time.time() - start_time))

df = pd.read_csv("../dataSets/congress_voting/CongressionalVotingID.shuf.test.csv", index_col="ID")
df = df.replace(["y", "n", "unknown"], ["1", "0", np.nan])
df.fillna(df.mode().iloc[0], inplace=True)
# Predict Output
predicted = model.predict(df)

df = pd.Series(predicted, index=df.index)

counter = len([name for name in os.listdir("results") if os.path.isfile(os.path.join("results", name))])
df.to_csv("results/result"+str(counter)+".csv", header=['class'])
