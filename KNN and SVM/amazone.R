#Rcript for amazon data
library(e1071)
library(class)


am<- data.frame(amazonReviews_800_train)

#remove Id
am$ID<- NULL

a<- data.frame(am[,c(10001,1:10000)])
#train test
smp_size <- floor(0.80 * nrow(a))
set.seed(123)
train_ind <- sample(seq_len(nrow(a)), size = smp_size)

atrain <- a[train_ind, ]
atest <- a[-train_ind, ]


####mulitple knn models
knn <- knn(atrain[,2:10001],atest[,2:10001], atrain[,1], k = 1)
knn.1 <- knn(atrain[,2:10001],atest[,2:10001], atrain[,1], k = 3)
knn.2 <- knn(atrain[,2:10001],atest[,2:10001], atrain[,1], k = 5)
knn.3 <- knn(atrain[,2:10001],atest[,2:10001], atrain[,1], k = 7)
knn.4 <- knn(atrain[,2:10001],atest[,2:10001], atrain[,1], k = 19)
knn.5 <- knn(atrain[,2:10001],atest[,2:10001], atrain[,1], k = 21)
knn.6 <- knn(atrain[,2:10001],atest[,2:10001], atrain[,1], k = 23)
knn.7<- knn(atrain[,2:10001],atest[,2:10001], atrain[,1], k = 31)
knn.8 <- knn(atrain[,2:10001],atest[,2:10001], atrain[,1], k = 37)
knn.9 <- knn(atrain[,2:10001],atest[,2:10001], atrain[,1], k = 51)
knn.10 <- knn(atrain[,2:10001],atest[,2:10001], atrain[,1], k = 55)
knn.11 <- knn(atrain[,2:10001],atest[,2:10001], atrain[,1], k = 57)
knn.12 <- knn(atrain[,2:10001],atest[,2:10001], atrain[,1], k = 61)

knn.13 <- knn(atrain[,2:10001],atest[,2:10001], atrain[,1], k = 71)

knn.14 <- knn(atrain[,2:10001],atest[,2:10001], atrain[,1], k = 75)
knn.15 <- knn(atrain[,2:10001],atest[,2:10001], atrain[,1], k = 79)
knn.16 <- knn(atrain[,2:10001],atest[,2:10001], atrain[,1], k = 81)





#confusinon matrix knn
atab<- table(knn, atest$Class)
sum(diag(atab))/sum(atab)


atab1<- table(knn.1, atest$Class)
sum(diag(atab1))/sum(atab1)

atab2<- table(knn.2, atest$Class)
sum(diag(atab2))/sum(atab2)

atab3<- table(knn.3, atest$Class)
sum(diag(atab3))/sum(atab3)

atab4<- table(knn.4, atest$Class)
sum(diag(atab4))/sum(atab4)

atab5<- table(knn.5, atest$Class)
sum(diag(atab5))/sum(atab5)

atab6<- table(knn.6, atest$Class)
sum(diag(atab6))/sum(atab6)

atab7<- table(knn.7, atest$Class)
sum(diag(atab7))/sum(atab7)

atab8<- table(knn.8, atest$Class)
sum(diag(atab8))/sum(atab8)

atab9<- table(knn.9, atest$Class)
sum(diag(atab9))/sum(atab9)

atab10<- table(knn.10, atest$Class)
sum(diag(atab10))/sum(atab10)

atab11<- table(knn.11, atest$Class)
sum(diag(atab11))/sum(atab11)

atab12<- table(knn.12, atest$Class)
sum(diag(atab12))/sum(atab12)

atab13<- table(knn.13, atest$Class)
sum(diag(atab13))/sum(atab13)


atab14<- table(knn.14, atest$Class)
sum(diag(atab14))/sum(atab14)

atab15<- table(knn.15, atest$Class)
sum(diag(atab15))/sum(atab15)

atab16<- table(knn.16, atest$Class)
sum(diag(atab16))/sum(atab16)
#########multiple svm models

aasvm<- svm(as.factor(atrain$Class)~.,atrain, scale = FALSE)
asvm<- svm(as.factor(atrain$Class)~.,atrain)
asvm1<- svm(as.factor(atrain$Class)~., atrain, kernel = "linear")

asvm3<- svm(as.factor(atrain$Class)~., atrain, gamma = 0.02, cost = 2)
asvm4<- svm(as.factor(atrain$Class)~.,atrain, kernel = "linear", gamma = 0.02, cost = 2)

asvm5<- svm(as.factor(atrain$Class)~., atrain, gamma = 0.033, cost = 2)
asvm6<- svm(as.factor(atrain$Class)~., atrain, kernel = "linear", gamma = 0.033, cost = 2)

asvm7<- svm(as.factor(atrain$Class)~., atrain, gamma = 0.2, cost = 5)
asvm8<- svm(as.factor(atrain$Class)~., atrain, kernel = "linear",gamma = 0.2, cost = 5)

#atrain$Class<-as.factor(atrain$Class)
#x<- as.matrix(atrain[,2:10001])
#y <- factor(atrain$Class)
#tu<- tune.svm(x, y , gamma=10^(-6:-2), cost=10^(1:2), scale = FALSE)
#tu1<- tune.svm(x , y, gamma=10^(-6:-2), cost=10^(1:2), scale = FALSE, kernel = "linear") 

#tsvm<- svm(as.factor(atrain$Class)~., atrain, gamma = 0.00001, cost = 10)
#tsvm1<- svm(as.factor(atrain$Class)~., atrain, kernel = "linear",gamma = 0.000001, cost = 10)


#tup<- predict(tsvm, atest)
#tup1<- predict(tsvm1, atest)

#tut<- table(tup, atest$Class)
#sum(diag(tut))/sum(tut)  
#tut1<- table(tup1, atest$Class)
#sum(diag(tut1))/sum(tut1)
#print(tu1)


aasp<- predict(aasvm, atest)
asp<- predict(asvm, atest)
asp1<- predict(asvm1, atest)
asp3<- predict(asvm3, atest)
asp4<- predict(asvm4, atest)
asp5<- predict(asvm5, atest)
asp6<- predict(asvm6, atest)
asp7<- predict(asvm7, atest)
asp8<- predict(asvm8, atest)

aamt<- table(aasp, atest$Class)
sum(diag(aamt))/sum(aamt)

amt<- table(asp, atest$Class)
sum(diag(amt))/sum(amt)

amt1<- table(asp, atest$Class)
sum(diag(amt1))/sum(amt1)

amt3<- table(asp3, atest$Class)
sum(diag(amt3))/sum(amt3)

amt4<- table(asp4, atest$Class)
sum(diag(amt4))/sum(amt4)

amt5<- table(asp5, atest$Class)
sum(diag(amt5))/sum(amt5)

amt6<- table(asp6, atest$Class)
sum(diag(amt6))/sum(amt6)

amt7<- table(asp7, atest$Class)
sum(diag(amt7))/sum(amt7)

amt8<- table(asp8, atest$Class)
sum(diag(amt8))/sum(amt8)


#precisoon recal, and f measure
#for Svm
#cm = as.matrix(tut)
#precision = diag(cm) / colSums(cm) 

#recall = diag(cm) / rowSums(cm) 
#f1 = 2 * precision * recall / (precision + recall) 
#f1

#precisoon recal, and f measure
#for Knn
#cm1 = as.matrix(ktt4)
#precision1 = diag(cm1) / colSums(cm1) 
#recall1 = diag(cm1) / rowSums(cm1) 
#f11 = 2 * precision1 * recall1 / (precision1 + recall1) 
#f11

