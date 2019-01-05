library(rpart)
library(caret)
library(plyr)
library(ISLR)
library(splines)
library(mgcv)
library(dplyr)
library(imputeTS)
library(MASS)
library(DAAG)

set.seed(123)

train<-read.csv("C:/Users/goso/Desktop/házifeladatok/Machine Learning/assignment2/Auto-mpg/AutoMPG.shuf.train.csv",header=TRUE)
test<-read.csv("C:/Users/goso/Desktop/házifeladatok/Machine Learning/assignment2/Auto-mpg/AutoMPG.shuf.test.csv",header=TRUE)

train<-train[ , -which(names(train) %in% c("carName"))]
test<-test[ , -which(names(test) %in% c("carName"))]
train=na.mean(train)
test=na.mean(test)
train$horsepower=as.numeric(train$horsepower)
test$horsepower=as.numeric(test$horsepower)

linmod=lm(mpg~., data=train)
linmod=stepAIC(linmod, direction="backward")
summary(linmod)

model_lda_full<-lda(mpg~horsepower+weight+modelYear+origin, data=train)

pred<-predict(model_lda_full, newdata = test)
new<-cbind(test['id'],as.numeric(pred$class))
write.csv(new, file = "solution1_lda1.csv",quote=FALSE,row.names=FALSE)

#lda=3.2

#lets do something with glm():
?glm

model_glm_log=glm(mpg~., data=train, family = "poisson")

pred<-predict(model_glm_log, newdata = test)
new<-cbind(test['id'],exp(pred))
write.csv(new, file = "solution1_glm_log.csv",quote=FALSE,row.names=FALSE)

model_glm_log=glm(mpg~., data=train, family = "quasipoisson")

pred<-predict(model_glm_log, newdata = test)
new<-cbind(test['id'],exp(pred))
write.csv(new, file = "solution1_glm_quasilog.csv",quote=FALSE,row.names=FALSE)

#using a trimmed not overfitting model:
model_glm_log=glm(mpg~horsepower+weight+modelYear+origin, data=train, family = "quasipoisson")

pred<-predict(model_glm_log, newdata = test)
new<-cbind(test['id'],exp(pred))
write.csv(new, file = "solution1_glm_log_notfull.csv",quote=FALSE,row.names=FALSE)
