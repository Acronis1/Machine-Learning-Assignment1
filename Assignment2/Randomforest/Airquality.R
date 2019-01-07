library(corrplot)
library(randomForest)

air<- data.frame(AirQualityUCI)
corrplot(cor(air[3:15]))
summary(air)
names(air)
unique(air$Time)
class(air$Time)
#important variables selection based on forest model
#aa<- data.frame(air$Date, air$Time,air$NOx.GT., air$T, air$AH,air$PT08.S4.NO2.,
 #               air$RH,air$NO2.GT.,air$PT08.S1.CO.,air$CO.GT.)

#rmse(error)
#rmse <- function(error)
#{
#  sqrt(mean(error^2))
#}

# Function that returns Mean Absolute Error
#mae <- function(error)
#{
#  mean(abs(error))
#}




## 75% of the sample size
smp_size <- floor(0.75 * nrow(air))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(air)), size = smp_size)

train <- air[train_ind, ]
test <- air[-train_ind, ]

ar<- randomForest(train$CO.GT.~., train,importance=TRUE)
varImpPlot(ar)
varImpPlot(ar,type=2)
importance(ar)
ap<- predict(ar, test)

ar1<- randomForest(train$CO.GT.~., train, mtry= 6,importance=TRUE)
ar2<- randomForest(train$CO.GT.~., train, mtry= 7,importance=TRUE)
ar3<- randomForest(train$CO.GT.~., train, mtry= 8,importance=TRUE)
ar4<- randomForest(train$CO.GT.~., train, mtry= 9,importance=TRUE)
ar5<- randomForest(train$CO.GT.~., train, mtry= 10,importance=TRUE)
ar6<- randomForest(train$CO.GT.~., train, mtry= 11,importance=TRUE)
ar7<- randomForest(train$CO.GT.~., train, mtry= 11,ntree=800,importance=TRUE)
importance(ar7)
ar8<- randomForest(train$CO.GT.~., train, mtry= 11,ntree=1000,importance=TRUE)
ar9<- randomForest(train$CO.GT.~., train, mtry= 11,ntree=1000)


ap1<- predict(ar1, test)
ap2<- predict(ar2, test)
ap3<- predict(ar3, test)
ap4<- predict(ar4, test)
ap5<- predict(ar5, test)
ap6<- predict(ar6, test)
ap7<- predict(ar7, test)
ap8<- predict(ar8, test)
ap9<- predict(ar9, test)



error <- test$air.CO.GT. - ap
rmse(error)
mae(error)
dim(test)
oob.err=double(14)
test.err=double(14)

for(mtry in 1:14){
  rf=randomForest(train$CO.GT. ~., train,mtry=mtry,ntree=500, importance = TRUE) 
  oob.err[mtry] = rf$mse[500] #Error of all Trees fitted
  
  pred<-predict(rf,test) #Predictions on Test Set for each Tree
  test.err[mtry]= with(test, mean( (CO.GT. - pred)^2)) #Mean Squared Test Error
  
  cat(mtry," ") #printing the output to the console
  
}


test.err
oob.err

matplot(1:mtry , cbind(oob.err,test.err), pch=19 , col=c("red","blue"),type="b",ylab="Mean Squared Error",xlab="Number of Predictors Considered at each Split")
legend("topright",legend=c("Out of Bag Error","Test Error"),pch=19, col=c("red","blue"))

