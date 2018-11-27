from sklearn.naive_bayes import ComplementNB, MultinomialNB
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
import os
import time


df = pd.read_csv("../dataSets/amazon_reviews/amazonReviews.800.train.csv", index_col="ID")
print(len(df["Class"].value_counts()))
classes = df["Class"]
dfTrain = df.drop(columns=["Class"])
print(dfTrain.head())

# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(dfTrain)

# tf_transformer = TfidfTransformer(use_idf=False).fit(dfTrain)
# X_train_tf = tf_transformer.transform(dfTrain)

start_time = time.time()

model = MultinomialNB(alpha=0.1)
model.fit(dfTrain, classes)

print("--- %s seconds ---" % (time.time() - start_time))

dfTest = pd.read_csv("../dataSets/amazon_reviews/amazonReviews.700.test.csv", index_col="ID")

# X_Test_tfidf = tfidf_transformer.transform(dfTest)

# tf_transformer = TfidfTransformer(use_idf=False).fit(dfTest)
# X_Test_tf = tf_transformer.transform(dfTest)

predicted = model.predict(dfTest)

dfTest = pd.Series(predicted, index=dfTest.index)

#counter = len([name for name in os.listdir("results") if os.path.isfile(os.path.join("results", name))])
#df.to_csv("results/result"+str(counter)+".csv", header=['class'])
