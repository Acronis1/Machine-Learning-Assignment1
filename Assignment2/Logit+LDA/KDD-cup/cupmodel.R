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

train<-read.csv("C:/Users/goso/Desktop/házifeladatok/Machine Learning/assignment2/KDD-cup/cup98ID.shuf.5000.train.csv",header=TRUE)
test<-read.csv("C:/Users/goso/Desktop/házifeladatok/Machine Learning/assignment2/KDD-cup/cup98ID.shuf.5000.test.csv",header=TRUE)

train=na.mean(train, option = "mean")
test=na.mean(test, option = "mean")
train=train[, sapply(train, function(col) length(unique(col))) > 1]
test=test[, sapply(test, function(col) length(unique(col))) > 1]

train <- train[,colSums(is.na(train))<nrow(train)]
test <- test[,colSums(is.na(test))<nrow(test)]

train <- mutate_all(train, function(x) as.numeric(as.character(x)))
test <- mutate_all(test, function(x) as.numeric(as.character(x)))

#removing columns where all the observatuons are NA-s

train <- train[,colSums(is.na(train))<nrow(train)]
test <- test[,colSums(is.na(test))<nrow(test)]

train$TARGET_D=train$TARGET_D+1 # this is neccesary for logit, as log(0) is not defined
colnames(test)[which(names(test) == "ADATE_22")] <- "ADATE_2"
#variable selection

control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
?train
model <- train(TARGET_D~., data=train, method="glmnet", preProcess="scale", trControl=control, na.action=na.omit)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
#most important are:

#ETH13+LSC2+ETH5+OCC6+LFC7+EIC8+OCC5+EIC9+ETH15+ANC8+ETH16+EC7+HHAS2+EC3

?glm

model_glm_log=glm((TARGET_D)~ETH13+LSC2+ETH5+OCC6+LFC7+EIC8+OCC5+EIC9+ETH15+ANC8+ETH16+EC7+HHAS2+EC3, data=train, family="poisson")




pred<-predict(model_glm_log, newdata = test)
new<-cbind(test['CONTROLN'],pred-1)
new$`pred - 1`[new$`pred - 1` < 0] <- 0
new$`pred - 1`=exp(new$`pred - 1`)
write.csv(new, file = "solution1_glm_kdd.csv",quote=FALSE,row.names=FALSE)
train$TARGET_D
#RMSE=4



model_lda<-lda((TARGET_D)~ETH13+LSC2+ETH5+OCC6+LFC7+EIC8+OCC5+EIC9+ETH15+ANC8+ETH16+EC7+HHAS2+EC3, data=train)

pred<-predict(model_lda, newdata = test)

new<-cbind(test['CONTROLN'],as.numeric(pred$class)-1)

as.numeric(pred$class)-1
write.csv(new, file = "solution1_ldakdd.csv",quote=FALSE,row.names=FALSE)



