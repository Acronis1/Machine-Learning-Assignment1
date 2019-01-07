library(corrplot)
library(randomForest)

ca<- data.frame(cancer)
dim(ca)
table(as.numeric(ca$X32))
ca$X32[ca$X32 == "?"] <-0
ca$Outcome<- as.numeric(factor(ca$Outcome))
ca$X32<- as.numeric(factor(ca$X32))
ca$ID<-NULL
corrplot(cor(ca))
#ca$X5<-NULL
#ca$X19<-NULL
#ca$X17<-NULL
#ca$X21<-NULL
#ca$X14<-NULL
#ca$X26<-NULL
#ca$X15<-NULL
#ca$X20<-NULL
#ca$X6<-NULL
#ca$X16<-NULL
#ca$X31<-NULL
#ca$X7<-NULL
#ca$X8<-NULL
## 75% of the sample size
smp_size <- floor(0.75 * nrow(ca))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(ca)), size = smp_size)

train <- ca[train_ind, ]
test <- ca[-train_ind, ]
dim(train)
cr1<- randomForest(train$Time~.,mtry=20, ntree = 1000,train,importance=TRUE)
cr2<- randomForest(train$Time~.,mtry=30, ntree = 700,train,importance=TRUE)
cr3<- randomForest(train$Time~.,mtry=31, ntree = 900,train,importance=TRUE)
cr4<- randomForest(train$Time~.,mtry=11, ntree = 1500,train,importance=TRUE)
cr5<- randomForest(train$Time~.,mtry=7, ntree = 1000,train,importance=TRUE)
cr6<- randomForest(train$Time~.,mtry=6, ntree = 800,train,importance=TRUE)
cr7<- randomForest(train$Time~.,mtry=5, ntree = 1100,train,importance=TRUE)
cr8<- randomForest(train$Time~.,mtry=9, ntree = 1500,train,importance=TRUE)
cr9<- randomForest(train$Time~.,mtry=27, ntree = 1500,train,importance=TRUE)
cr10<- randomForest(train$Time~.,train,importance=TRUE)
cr11<- randomForest(train$Time~.,ntree = 50,train,importance=TRUE)
cr12<- randomForest(train$Time~.,ntree = 200,train,importance=TRUE)
varImpPlot(cr)

cp1<- predict(cr1, test)
cp2<- predict(cr2, test)
cp3<- predict(cr3, test)
cp4<- predict(cr4, test)
cp5<- predict(cr5, test)
cp6<- predict(cr6, test)
cp7<- predict(cr7, test)
cp8<- predict(cr8, test)
cp9<- predict(cr9, test)
cp10<- predict(cr10, test)
cp11<- predict(cr11, test)
cp12<- predict(cr12, test)

error <- test$Time - cp12
rmse(error)
mae(error)

rmse <- function(error)
{
  sqrt(mean(error^2))
}

# Function that returns Mean Absolute Error
mae <- function(error)
{
  mean(abs(error))
}



oob.err=double(33)
test.err=double(33)

for(mtry in 1:33){
  rf=randomForest(train$Time ~., train,mtry=mtry,ntree=500, importance = TRUE) 
  oob.err[mtry] = rf$mse[500] #Error of all Trees fitted
  
  pred<-predict(rf,test) #Predictions on Test Set for each Tree
  test.err[mtry]= with(test, mean( (Time - pred)^2)) #Mean Squared Test Error
  
  cat(mtry," ") #printing the output to the console
  
}


test.err
oob.err

matplot(1:mtry , cbind(oob.err,test.err), pch=19 , col=c("red","blue"),type="b",ylab="Mean Squared Error",xlab="Number of Predictors Considered at each Split")
legend("topright",legend=c("Out of Bag Error","Test Error"),pch=19, col=c("red","blue"))

