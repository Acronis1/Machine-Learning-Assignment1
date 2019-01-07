library(corrplot)
library(randomForest)
auto<- data.frame(AutoMPG_shuf_train)
summary(auto)

auto$horsepower[auto$horsepower == "?"] <-106.63
auto$horsepower<- as.numeric(auto$horsepower)
auto$carName<- as.numeric(as.factor(auto$carName))
auto$id<- NULL
corrplot(cor(auto))
## 75% of the sample size
smp_size <- floor(0.75 * nrow(auto))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(auto)), size = smp_size)

train <- auto[train_ind, ]
test <- auto[-train_ind, ]
#tr<- data.frame(train$weight,train$displacement,train$horsepower,train$modelYear,train$carName, train$mpg)
#te<- data.frame(test$weight,test$displacement,test$horsepower,test$modelYear,test$carName, test$mpg)
summary(train)
library(randomForest)
r2<- randomForest(train$mpg~., train,ntree=500,mtry = 2,importance = TRUE)
r3<- randomForest(train$mpg~., train,ntree=500,mtry = 3,importance = TRUE)
r4<- randomForest(train$mpg~., train,ntree=500,mtry = 4,importance = TRUE)
r5<- randomForest(train$mpg~., train,ntree=500,mtry = 5,importance = TRUE)
r6<- randomForest(train$mpg~., train,ntree=500,mtry = 6,importance = TRUE)
r7<- randomForest(train$mpg~., train,ntree=500,mtry = 7,importance = TRUE)
r8<- randomForest(train$mpg~., train,ntree=500,mtry = 8,importance = TRUE)

r62<- randomForest(train$mpg~., train,ntree=200,mtry = 6,importance = TRUE)
r66<- randomForest(train$mpg~., train,ntree=600,mtry = 6,importance = TRUE)
r67<- randomForest(train$mpg~., train,ntree=700,mtry = 6,importance = TRUE)


p2<- predict(r2, test)
p3<- predict(r3, test)
p4<- predict(r4, test)
p5<- predict(r5, test)
p6<- predict(r6, test)
p7<- predict(r7, test)
p8<- predict(r8, test)

p62<- predict(r62, test)
p66<- predict(r66, test)
p67<- predict(r67, test)


error <- test$mpg - p67
rmse(error)
mae(error)



oob.err=double(8)
test.err=double(8)

#mtry is no of Variables randomly chosen at each split
for(mtry in 1:8){
  rf=randomForest(train$mpg ~., train,mtry=mtry,ntree=500, importance = TRUE) 
  oob.err[mtry] = rf$mse[500] #Error of all Trees fitted
  
  pred<-predict(rf,test) #Predictions on Test Set for each Tree
  test.err[mtry]= with(test, mean( (mpg - pred)^2)) #Mean Squared Test Error
  
  cat(mtry," ") #printing the output to the console
  
}


test.err
oob.err

matplot(1:mtry , cbind(oob.err,test.err), pch=19 , col=c("red","blue"),type="b",ylab="Mean Squared Error",xlab="Number of Predictors Considered at each Split")
legend("topright",legend=c("Out of Bag Error","Test Error"),pch=19, col=c("red","blue"))


####R squared
rss <- sum((p - test$mpg) ^ 2)  ## residual sum of squares
tss <- sum((test$mpg - mean(test$mpg)) ^ 2)  ## total sum of squares
rsq <- 1 - rss/tss

# Function that returns Root Mean Squared Error
error <- test$mpg - p
rmse(error)
rmse <- function(error)
{
  sqrt(mean(error^2))
}

# Function that returns Mean Absolute Error
mae <- function(error)
{
  mean(abs(error))
}




