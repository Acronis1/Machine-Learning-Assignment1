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

data <- read.csv("C:/Users/goso/Desktop/házifeladatok/Machine Learning/assignment2/Air_quality/AirQualityUCI.csv", header=TRUE, sep=";")
data = data[ , -which(names(data) %in% c("X","X.1"))]
data$CO.GT.=as.numeric(data$CO.GT.)
data$C6H6.GT.=as.numeric(data$C6H6.GT.)

data=na_if(data, -200)
data=na_if(data, 2)
data=na_if(data, "")


#I don't want to fill in missing labels, it would probably introduce bias, still we have 7765 
#observations, it should be sufficient.
data=data[!is.na(data$CO.GT.),]
#filling in with means.
data=na.mean(data)


#basic data exploration:

plot(train)

#it is visible that date, time T, Ah, RH have no correlation with our label, let's omit them:
data = data[ , -which(names(data) %in% c("T","RH","AH","Date","Time"))]

#train-test split, 80-20%:

smp_size <- floor(0.8 * nrow(data))
train_ind <- sample(seq_len(nrow(data)), size = smp_size)
train<-data[train_ind,]
test<-data[-train_ind,]

#building logit model:
?glm
model_glm_log=glm(CO.GT.~., data=train, family = "poisson", na.action=na.exclude)

pred_glm<-predict(model_glm_log, newdata = test)

#let's calculate the RMSE:
rmse_log = sqrt(mean((pred_glm-test$CO.GT.)^2))
rmse_log
#given the range of the label 1-105, this is a rather poor score.
#rmse = 29,45



#using LDA:

model_lda<-lda(CO.GT.~., data=train)

pred_lda<-as.numeric(predict(model_lda, newdata = test)$class)

rmse_lda = sqrt(mean((pred_lda-test$CO.GT.)^2))
rmse_lda
#significantly better results using lda, rmse = 8,2822


#basic feature selection:
linmod=lm(CO.GT.~., data=train)
linmod=stepAIC(linmod, direction="backward")
summary(linmod)

#let's build the reduced models:

model_glm_log_red=glm(CO.GT.~PT08.S1.CO.+NMHC.GT.+PT08.S2.NMHC.+NOx.GT.+PT08.S3.NOx.+NO2.GT.+PT08.S4.NO2.+PT08.S5.O3., data=train, family = "poisson", na.action=na.exclude)
summary(model_glm_log_red)

pred_glm_red<-predict(model_glm_log_red, newdata = test)

#let's calculate the RMSE:
rmse_log_red = sqrt(mean((pred_glm_red-test$CO.GT.)^2))
rmse_log_red
#there is so significant difference to the full model


model_lda<-lda(CO.GT.~PT08.S1.CO.+NMHC.GT.+PT08.S2.NMHC.+NOx.GT.+PT08.S3.NOx.+NO2.GT.+PT08.S4.NO2.+PT08.S5.O3., data=train)

pred_lda<-as.numeric(predict(model_lda, newdata = test)$class)

rmse_lda = sqrt(mean((pred_lda-test$CO.GT.)^2))
rmse_lda

#we could achieve a slightly lower RMSE, namely rmse = 7,979