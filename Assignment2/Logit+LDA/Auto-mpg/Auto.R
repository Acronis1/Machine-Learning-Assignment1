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
na.mean(train)
na.mean(test)
train2 <- mutate_all(train, function(x) as.numeric(as.character(x)))
test2 <- mutate_all(test, function(x) as.numeric(as.character(x)))

train2<-na.mean(train2)
test2<-na.mean(test2)



mod.gam2=gam(log(mpg)~s(cylinders,k=3,bs="cr")+s(displacement,k=5,bs="cr")+s(horsepower,k=5,bs="cr")+s(weight,k=5,bs="cr")+s(acceleration,k=5,bs="cr")+s(modelYear,k=5,bs="cr")+origin, data=train2)
pred<-predict(mod.gam2, newdata = test2)
pred=exp(pred)

new<-cbind(test['id'],pred)
write.csv(new, file = "solution2_splines.csv",quote=FALSE,row.names=FALSE)


model2<-lm(mpg~ns(cylinders,knots=3)+ns(displacement,knots=3)+ns(horsepower,knots=3)+ns(weight,knots=3)+ns(acceleration,knots=3)+ns(modelYear,knots=3)+origin, data=train2)
pred<-predict(model2, newdata = test2)
pred

new<-cbind(test['id'],pred)
write.csv(new, file = "solution1_nsplines.csv",quote=FALSE,row.names=FALSE)


cv.lm(model2)
CVlm(data = train2, form.lm = formula(mpg ~ .-id), m=3, dots = 
        FALSE, seed=29, plotit=TRUE, printit=TRUE)

#experimenting with cv:

smp_size <- floor(0.8 * nrow(train))
train_ind <- sample(seq_len(nrow(train2)), size = smp_size)
training<-train2[train_ind,]
testing<-train2[-train_ind,]

###############################################

fitControl <- trainControl(method = "cv", number = 10)

fit <- train(Purchase ~ ., 
             data = training, 
             method = "lm", 
             trControl = fitControl)

###############################################

fits <- list()
for(i in 2:11){
  fitControl <- trainControl(method = "cv", number = i)
  
  fits[[i]] <- train(mpg ~horsepower , 
                     data = training, 
                     method = "lm", 
                     trControl = fitControl)
}
summary(fits[10])

testing2<-testing[ , -which(names(testing) %in% c("mpg"))]

pred1<-predict(fits[3],  newdata = testing2)
pred2<-predict(fits[10],  newdata = testing2)
pred1
pred2

?predict
mse<-list()

testing['pred']=predict(fits[18],  newdata = testing)
mse=mean((testing$mpg-testing$pred)^2)
mse

for(i in 2:50){
  testing['pred']=predict(fits[[i]], x=testing$mpg, newdata = testing)
  
  mse[[i]]=mean((testing$mpg-testing$pred)^2)
  
}
y=unlist(mse)
plot(y,xlab="nr of k in cv",ylab="MSE values")


