#library(liquidSVM)
library(e1071)
library(class)
#remove.packages("liquidSVM")
ct<- data.frame(covtype_data)
dim(ct)
summary(ct)
table(ct$Cover_Type)

ct1<- subset(ct, ct$Cover_Type == "1")
ct2<- subset(ct, ct$Cover_Type == "2")
ct3<- subset(ct, ct$Cover_Type >= "3")


######## dataset was huge, and SVM was taking to long to run, so 
#we reducde the 1 and 2 class which are the hugge majority compare to other class
###### covertype 1

#'taking covertpe 1 only 10%data'
smp_size1 <- floor(0.15 * nrow(ct1))
set.seed(123)
train_ind1 <- sample(seq_len(nrow(ct1)), size = smp_size1)

ctt1 <- ct1[train_ind1, ]
test1 <- ct1[-train_ind1, ]

dim(ctt1)


###### covertype 2  only 10%data'
smp_size1 <- floor(0.10 * nrow(ct2))
set.seed(123)
train_ind1 <- sample(seq_len(nrow(ct2)), size = smp_size1)

ctt2 <- ct2[train_ind1, ]

dim(ctt2)
#combine all three datasets
c<- rbind(ctt1, ctt2,ct3)


#scling function
normalize <- function(x){
  return ((x-min(x))/(max(x)-min(x)))
}


c1<- as.data.frame(lapply(c[,c(seq(1:54))],normalize))
cc<- cbind(c1, c[,55])

dim(cc)


####train n test with scaled data
smp_size1 <- floor(0.70 * nrow(cc))
set.seed(123)
train_ind1 <- sample(seq_len(nrow(cc)), size = smp_size1)

ctrain <- cc[train_ind1, ]
ctest <- cc[-train_ind1, ]
ctrain.t<- ctrain[,55]
ctest.t<- ctest[,55]
dim(ctrain)

View(ctrain)
table(ctrain$Cover_Type)
####multiple modes Knn
knn.1 <-  knn(ctrain, ctest, ctrain.t, k=1)
knn.5 <-  knn(ctrain, ctest, ctrain.t, k=5)
knn.20 <- knn(ctrain, ctest, ctrain.t, k=21)



# without scaling train rtest
c<- rbind(ctt1, ctt2,ct3)
View(c)
c<- data.frame(c)
smp_size1 <- floor(0.70 * nrow(c))
set.seed(123)
train_ind1 <- sample(seq_len(nrow(c)), size = smp_size1)

c1train <- c[train_ind1, ]
c1test <- c[-train_ind1, ]
c1train.t<- c1train[,55]
c1test.t<- c1test[,55]
dim(ctrain)

#mutilple SVM models

#csvm<- svm(as.factor(ctrain.t)~.,ctrain)

csvm1<- svm(as.factor(c1train.t)~.,c1train )

csvm2<- svm(as.factor(c1train.t)~.,c1train , kernel = "linear")

#with scaling data
csvm3<- svm(as.factor(ctrain.t)~.,ctrain )

csvm4<- svm(as.factor(ctrain.t)~.,ctrain , kernel = "linear")

csvm5<- svm(as.factor(c1train.t)~.,c1train, gamma = 0.33, cost = 5 )

csvm6<- svm(as.factor(c1train.t)~.,c1train , kernel = "linear", gamma = 0.33, cost = 5)

csvm7<- svm(as.factor(ctrain.t)~.,ctrain,gamma = 0.33, cost = 5 )

csvm8<- svm(as.factor(ctrain.t)~.,ctrain , kernel = "linear", gamma = 0.33, cost = 5)


csvm9<- svm(as.factor(c1train.t)~.,c1train, gamma = 0.50, cost = 2 )

csvm10<- svm(as.factor(c1train.t)~.,c1train , kernel = "linear", gamma = 0.50, cost = 2 )

csvm11<- svm(as.factor(ctrain.t)~.,ctrain,gamma = 0.50, cost = 2  )

csvm12<- svm(as.factor(ctrain.t)~.,ctrain , kernel = "linear", gamma = 0.50, cost = 2 )





#Multiple svm predictions, confusion matix and accuracy
csvp1<- predict(csvm1, c1test)
tab1<- table(csvp1, c1test.t)
sum(diag(tab1))/sum(tab1)


csvp2<- predict(csvm2, c1test)
tab2<- table(csvp2, c1test.t)
sum(diag(tab2))/sum(tab2)

csvp3<- predict(csvm3, ctest)
tab3<- table(csvp3, ctest.t)
sum(diag(tab3))/sum(tab3)

csvp4<- predict(csvm4, ctest)
tab1<- table(csvp4, ctest.t)
sum(diag(tab1))/sum(tab1)


csvp5<- predict(csvm5, ctest)
tab5<- table(csvp5, c1test.t)
sum(diag(tab5))/sum(tab5)

csvp6<- predict(csvm6, ctest)
tab6<- table(csvp6, c1test.t)
sum(diag(tab6))/sum(tab6)

csvp7<- predict(csvm7, ctest)
tab7<- table(csvp7, c1test.t)
sum(diag(tab7))/sum(tab7)

csvp8<- predict(csvm8, ctest)
tab8<- table(csvp8, c1test.t)
sum(diag(tab8))/sum(tab8)

csvp9<- predict(csvm9, ctest)
tab9<- table(csvp9, c1test.t)
sum(diag(tab9))/sum(tab9)

csvp10<- predict(csvm10, ctest)
tab10<- table(csvp10, c1test.t)
sum(diag(tab10))/sum(tab10)

csvp11<- predict(csvm11, ctest)
tab11<- table(csvp11, c1test.t)
sum(diag(tab11))/sum(tab11)


csvp12<- predict(csvm12, ctest)
tab12<- table(csvp12, c1test.t)
sum(diag(tab12))/sum(tab12)

#mutiple knn models
knnc1<- knn(ctrain[,1:54], ctest[,1:54], ctrain$Cover_Type, k = 3)
knnc2<- knn(ctrain[,1:54], ctest[,1:54], ctrain$Cover_Type, k = 5)
knnc3<- knn(ctrain[,1:54], ctest[,1:54], ctrain$Cover_Type, k = 7)
knnc4<- knn(ctrain[,1:54], ctest[,1:54], ctrain$Cover_Type, k = 11)
knnc5<- knn(ctrain[,1:54], ctest[,1:54], ctrain$Cover_Type, k = 21)
knnc6<- knn(ctrain[,1:54], ctest[,1:54], ctrain$Cover_Type, k = 23)
knnc7<- knn(ctrain[,1:54], ctest[,1:54], ctrain$Cover_Type, k = 27)
knnc8<- knn(c1train[,1:54], c1test[,1:54], c1train$Cover_Type, k = 3)
#confusion matrix
kt1<- table(knnc1, ctest$Cover_Type)

kt2<- table(knnc2, ctest$Cover_Type)

kt3<- table(knnc3, ctest$Cover_Type)

kt4<- table(knnc4, ctest$Cover_Type)
kt5<- table(knnc5, ctest$Cover_Type)
kt6<- table(knnc6, ctest$Cover_Type)
kt7<- table(knnc7, ctest$Cover_Type)
kt8<- table(knnc8, ctest$Cover_Type)

#accuracy
sum(diag(kt1))/sum(kt1)
sum(diag(kt2))/sum(kt2)
sum(diag(kt3))/sum(kt3)
sum(diag(kt4))/sum(kt4)
sum(diag(kt5))/sum(kt5)
sum(diag(kt6))/sum(kt6)
sum(diag(kt7))/sum(kt7)
sum(diag(kt8))/sum(kt8)


#Preccisoon recal and f measure
cm = as.matrix(kt2)
precision <- diag(cm)/ colSums(cm)
recall<- diag(cm)/ rowSums(cm)
f1<- 2 * precision * recall/(precision+recall)


cm1 = as.matrix(tab11)
precision <- diag(cm)/ colSums(cm)
recall<- diag(cm)/ rowSums(cm)
f1<- 2 * precision * recall/(precision+recall)
