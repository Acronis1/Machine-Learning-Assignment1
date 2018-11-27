library(rpart)

setwd("C:/Users/goso/Desktop/házifeladatok/Machine Learning/assignment1/data amazon review")

datatrain<-read.csv(file="amazonReviews.800.train.csv",header = TRUE)
datatest<-read.csv(file="amazonReviews.700.test.csv",header = TRUE)

tree_model<-rpart(Class~., data=datatrain, method="class",control=rpart.control(minsplit=20, cp=0.001) ) 


prediction<-predict(tree_model, datatest, type = "class")

prediction

new<-cbind(datatest['ID'],prediction)

new
write.csv(new, file = "solution4.csv",quote=FALSE,row.names=FALSE)
