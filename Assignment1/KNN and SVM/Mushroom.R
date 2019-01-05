
library(e1071)
library(class)

m<- data.frame(mus)

length(which(m$stalk.root == '?'))
length(which(m == '?'))
#replacing  ? with majority
m$stalk.root<- gsub("?", "b", m$stalk.root, fixed = TRUE)

m1<-m[, 2:23]
View(m)
dim(m)
m1[] <- lapply(m1, function(x) as.numeric(as.factor(x)))
m1<- cbind(m$classes, m1)


colnames(m1)[1] <- "classes"
m1$classes<- as.numeric(as.factor(m1$classes))
names(m1)
## 75% of the sample size
smp_size <- floor(0.75 * nrow(m1))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(m1)), size = smp_size)

train <- m1[train_ind, ]
test <- m1[-train_ind, ]

###multiple SVM models
ms<- svm(as.factor(classes)~., train, scale = FALSE)

ms1<- svm(as.factor(classes)~., train, scale = FALSE, kerne = "linear")


ms2<- svm(as.factor(classes)~., train, gamma = 0.33, cost = 1)

ms3<- svm(as.factor(classes)~., train, kerne = "linear", gamma = 0.33, cost = 1)


ms4<- svm(as.factor(classes)~., train, scale = FALSE, gamma = 0.05, cost = 2)

ms5<- svm(as.factor(classes)~., train, scale = FALSE, kerne = "linear", gamma = 0.05, cost = 2)


ms6<- svm(as.factor(classes)~., train, gamma = 0.0001, cost = 3)

ms7<- svm(as.factor(classes)~., train, kerne = "linear", gamma = 0.0001, cost = 3)



#mtu<- tune.svm(train[,2:23], as.factor(train$classes) , gamma=10^(-6:-1), cost=10^(1:2))
ms8<- svm(as.factor(classes)~., train, kerne = "linear", gamma = 0.01, cost = 10)
#mtu1<- tune.svm(train[,2:23], as.factor(train$classes), gamma=10^(-6:-1), cost=10^(1:2), kernel = "linear") 
ms9<- svm(as.factor(classes)~., train, kerne = "linear", gamma = 0.0000001, cost = 10)
 

#Svm predictions
msp<- predict(ms, test)
msp1<- predict(ms1, test)
msp2<- predict(ms2, test)
msp3<- predict(ms3, test)
msp4<- predict(ms4, test)
msp5<- predict(ms5, test)
msp6<- predict(ms6, test)
msp7<- predict(ms7, test)
msp8<- predict(ms8, test)
msp9<- predict(ms9, test)
#'muliple comfusin matix'
mtab<- table(msp, test$classes)
sum(diag(mtab))/sum(mtab)

mtab1<- table(msp1, test$classes)
sum(diag(mtab1))/sum(mtab1)

mtab2<- table(msp2, test$classes)
sum(diag(mtab2))/sum(mtab2)

mtab3<- table(msp3, test$classes)
sum(diag(mtab3))/sum(mtab3)

mtab4<- table(msp4, test$classes)
sum(diag(mtab4))/sum(mtab4)

mtab5<- table(msp5, test$classes)
sum(diag(mtab5))/sum(mtab5)

mtab6<- table(msp6, test$classes)
sum(diag(mtab6))/sum(mtab6)

mtab7<- table(msp7, test$classes)
sum(diag(mtab7))/sum(mtab7)

mtab8<- table(msp8, test$classes)
sum(diag(mtab8))/sum(mtab8)

mtab9<- table(msp9, test$classes)
sum(diag(mtab9))/sum(mtab9)


#With scale
smp_size <- floor(0.75 * nrow(m1))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(m1)), size = smp_size)


ktrain <- m1[train_ind, ]

ktest<- test <- m1[-train_ind, ]

##knn models
knn.3 <-  knn(ktrain[,2:23], ktest[,2:23], ktrain$classes, k=3)
knn.5 <-  knn(ktrain[,2:23], ktest[,2:23], ktrain$classes, k=5)
knn.7 <-  knn(ktrain[,2:23], ktest[,2:23], ktrain$classes, k=7)
knn.11 <-  knn(ktrain[,2:23], ktest[,2:23], ktrain$classes, k=11)
knn.17 <-  knn(ktrain[,2:23], ktest[,2:23], ktrain$classes, k=17)

knn.21 <- knn(ktrain[,2:23], ktest[,2:23], ktrain$classes, k=21)
knn.23 <- knn(ktrain[,2:23], ktest[,2:23], ktrain$classes, k=23)
knn.31 <- knn(ktrain[,2:23], ktest[,2:23], ktrain$classes, k=31)
knn.27 <- knn(ktrain[,2:23], ktest[,2:23], ktrain$classes, k=27)
knn.57 <- knn(ktrain[,2:23], ktest[,2:23], ktrain$classes, k=57)

100 * sum(ktrain.t == knn.1)/100


knt3<- table(knn.3 ,ktest$classes)
sum(diag(knt3))/sum(knt3)

knt5<- table(knn.5 ,ktest$classes)
sum(diag(knt5))/sum(knt5)
knt7<- table(knn.7 ,ktest$classes)
sum(diag(knt7))/sum(knt7)
knt11<- table(knn.11 ,ktest$classes)
sum(diag(knt11))/sum(knt11)
knt17<- table(knn.17 ,ktest$classes)
sum(diag(knt17))/sum(knt17)


knt21<- table(knn.21 ,ktest$classes)
sum(diag(knt21))/sum(knt21)
knt23<- table(knn.23 ,ktest$classes)
sum(diag(knt23))/sum(knt23)
knt31<- table(knn.31 ,ktest$classes)
sum(diag(knt31))/sum(knt31)
knt27<- table(knn.27 ,ktest$classes)
sum(diag(knt27))/sum(knt27)
knt57<- table(knn.57 ,ktest$classes)
sum(diag(knt57))/sum(knt57)




#precison, recall, adm f measure of best models



#for svm
cm = as.matrix(mtab3)
precision = diag(cm) / colSums(cm) 
recall = diag(cm) / rowSums(cm) 
f1 = 2 * precision * recall / (precision + recall) 
f1


#for knn
cm1 = as.matrix(knt11)
precision1 = diag(cm1) / colSums(cm1) 
recall1 = diag(cm1) / rowSums(cm1) 
f11 = 2 * precision1 * recall1 / (precision1 + recall1) 
f11




