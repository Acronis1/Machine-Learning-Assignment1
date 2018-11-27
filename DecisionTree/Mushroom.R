library(rpart)
library(caret)

set.seed(123)
data<-read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",header=FALSE)
names(data)<- c("class","cap-shape","cap-surface","cap-color","bruises","odor","gill-attachment","gill-spacing","gill-size","gill-color","stalk-shape","stalk-root","stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring","veil-type","veil-color","ring-number","ring-type","spore-print-color","population","habitat")


smp_size <- floor(0.8 * nrow(data))
train_ind <- sample(seq_len(nrow(data)), size = smp_size)
train<-data[train_ind,]
test<-data[-train_ind,]



tree_model<-rpart(class~., data=train, method="class",control=rpart.control(minsplit=20, cp=0.01) ) 


prediction<-predict(tree_model, test, type = "class")
confusionMatrix(prediction,test$class)

##pretty good, false negatives:9, false positives 0


tree_model<-rpart(class~., data=train, method="class",control=rpart.control(minsplit=2, cp=0.001) ) 


prediction<-predict(tree_model, test, type = "class")
confusionMatrix(prediction,test$class)

#Even better with overfitting ! Data is biased?