library(corrplot)
library(randomForest)


cu<- data.frame(cup98ID_shuf_5000_train)
View(cu)
#?apply()
#cu<- data.frame(apply(cu, 2,function(x) as.numeric(factor(x))))
c<- data.frame()

cu1<- data.frame(cu$TARGET_D,cu$ETH13,cu$LSC2,cu$ETH5,cu$OCC6,cu$LFC7,
                 cu$EIC8,cu$OCC5,cu$EIC9,
                     cu$ETH15,cu$ANC8,cu$ETH16,cu$EC7,cu$HHAS2,cu$EC3)


table(is.na(cu1))
View(cu1)
#cu[is.na(cu)]<- 1000



smp_size <- floor(0.75 * nrow(cu1))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(cu1)), size = smp_size)

train <- cu1[train_ind, ]
test <- cu1[-train_ind, ]

library(randomForest)
#rr1<- randomForest(cu$TARGET_D~cu$ETH13+cu$LSC2+cu$ETH5+cu$OCC6+cu$LFC7+cu$EIC8+cu$OCC5+cu$EIC9+
 #                    cu$ETH15+cu$ANC8+cu$ETH16+cu$EC7+cu$HHAS2+cu$EC3, data = cu)

rr1<- randomForest(train$cu.TARGET_D~.,train, importance=TRUE)
rr2<- randomForest(train$cu.TARGET_D~.,train,importance=TRUE, ntree= 1000)
rr3<- randomForest(train$cu.TARGET_D~.,train,importance=TRUE, ntree= 1000,mtry =12)
rr4<- randomForest(train$cu.TARGET_D~.,train,importance=TRUE, ntree= 500,mtry =11)
rr5<- randomForest(train$cu.TARGET_D~.,train,importance=TRUE, ntree= 700,mtry =7)
rr6<- randomForest(train$cu.TARGET_D~.,train,importance=TRUE, ntree= 800,mtry =5)
rr7<- randomForest(train$cu.TARGET_D~.,train,importance=TRUE, ntree= 700,mtry =9)
rr8<- randomForest(train$cu.TARGET_D~.,train,importance=TRUE, ntree= 1000,mtry =13)

rr9<- randomForest(train$cu.TARGET_D~.,train,importance=TRUE, ntree= 200,mtry =13)
rr10<- randomForest(train$cu.TARGET_D~.,train,importance=TRUE, ntree= 100)

rr11<- randomForest(train$cu.TARGET_D~.,train,importance=TRUE, ntree= 50)


pp1<- predict(rr1, test)
pp2<- predict(rr2, test)
pp3<- predict(rr3, test)
pp4<- predict(rr4, test)
pp5<- predict(rr5, test)
pp6<- predict(rr6, test)
pp7<- predict(rr7, test)
pp8<- predict(rr8, test)
pp9<- predict(rr9, test)
pp10<- predict(rr10, test)
pp11<- predict(rr11, test)


error <- cu$TARGET_D - pp11
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


oob.err=double(14)
test.err=double(14)

for(mtry in 1:14){
  rf=randomForest(train$cu.TARGET_D ~., train,mtry=mtry,ntree=500, importance = TRUE) 
  oob.err[mtry] = rf$mse[500] #Error of all Trees fitted
  
  pred<-predict(rf,test) #Predictions on Test Set for each Tree
  test.err[mtry]= with(test, mean( (cu.TARGET_D - pred)^2)) #Mean Squared Test Error
  
  cat(mtry," ") #printing the output to the console
  
}


test.err
oob.err

matplot(1:mtry , cbind(oob.err,test.err), pch=19 , col=c("red","blue"),type="b",ylab="Mean Squared Error",xlab="Number of Predictors Considered at each Split")
legend("topright",legend=c("Out of Bag Error","Test Error"),pch=19, col=c("red","blue"))

