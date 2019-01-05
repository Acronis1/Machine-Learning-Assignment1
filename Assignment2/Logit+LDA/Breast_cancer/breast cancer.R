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

data <- read.csv("C:/Users/goso/Desktop/házifeladatok/Machine Learning/assignment2/Breast_cancer/wpbc.data", header=FALSE, sep=",")
colnames(data)[1] = "id"
colnames(data)[2] = "label"

data$label=as.numeric(data$label)
data$V3=as.numeric(data$V3)
data=na_if(data, '?')
data[,35] = as.numeric(data[,35]) #column 35 was not numeric

data=na.mean(data)

#splitting data into train and test:

smp_size <- floor(0.8 * nrow(data))
train_ind <- sample(seq_len(nrow(data)), size = smp_size)
train<-data[train_ind,]
test<-data[-train_ind,]

#based on the high number of atriutes, let's do feature selection:
linmod=lm(label~., data=train)
linmod=stepAIC(linmod, direction="backward")
#here are the singificant attributes:
summary(linmod)

model_log=lm(label~.-id, data=data, na.action=na.exclude)

pred_log<-exp(predict(model_log, newdata = test))

#let's calculate the RMSE:
rmse_log = sqrt(mean((pred_log-test$label)^2))
rmse_log
#pretty bad result given that the label has two levels 1 and 2.
#rmse = 2.48

#let's do LDA, but first the number of variables must be reduced for lda,
#there are only 40 observatoions in the train set.

model_lda<-lda(label~V3+V4+V5+V7+V8+V13+V15+V16+V17+V19+V20+V24+V29+V30+V31+V34+V35, data=train)

pred_lda<-as.numeric(predict(model_lda, newdata = test)$class)

rmse_lda = sqrt(mean((pred_lda-test$label)^2))
rmse_lda
#significatnly better result from logit regression, but still the rmse is quite high
#given  that the label has only two values with 1 and 2
#rmse = 0.5244