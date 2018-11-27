library(rpart)
library(caret)
library(plyr)
set.seed(123)

train<-read.csv("C:/Users/goso/Desktop/házifeladatok/Machine Learning/assignment1/data congress voting/Congressional_train.csv",header=TRUE)
test<-read.csv("C:/Users/goso/Desktop/házifeladatok/Machine Learning/assignment1/data congress voting/Congressional_test.csv",header=TRUE)

aggdata<-aggregate(train, by=list(class),FUN=length)
count(train, 'class')

control <- trainControl(method="repeatedcv", number=10, repeats=3)
model <- train(class~., data=train, method="lvq", preProcess="scale", trControl=control)
importance <- varImp(model, scale=FALSE)
importance
colnames(test)[5]='a'
colnames(test)[4]='b'
colnames(test)[6]='c'
colnames(test)[13]='d'
colnames(test)[9]='e'
colnames(test)[15]='f'
colnames(test)[10]='g'
colnames(test)[14]='h'
colnames(test)[1]='ID'

colnames(train)[5]='a'
colnames(train)[4]='b'
colnames(train)[6]='c'
colnames(train)[13]='d'
colnames(train)[9]='e'
colnames(train)[15]='f'
colnames(train)[10]='g'
colnames(train)[14]='h'
colnames(train)[1]='ID'

tree_model<-rpart(class~a+b+c+d+e+f+g+h, data=train, method="class",na.action=na.pass) 
prediction<-predict(tree_model, test, type = "class")
prediction

new<-cbind(test['ID'],prediction)
write.csv(new, file = "solution6.csv",quote=FALSE,row.names=FALSE)
count(new, 'prediction')
