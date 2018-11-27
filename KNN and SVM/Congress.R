####rscript for congress voting 
library(e1071)
library(class)

v <-  read.csv("CongressionalVotingID_shuf_train")

#replacing all Unknown values with 2
v$handicapped.infants <- gsub("unknown", "2", v$handicapped.infants)

table(v$water.project.cost.sharing)
v$water.project.cost.sharing <- gsub("unknown", "2", v$water.project.cost.sharing)

table(v$adoption.of.the.budget.resolution)
v$adoption.of.the.budget.resolution <- gsub("unknown", "2", v$adoption.of.the.budget.resolution)

table(v$physician.fee.freeze)

v$physician.fee.freeze <- gsub("unknown", "2", v$physician.fee.freeze)

table(v$el.salvador.aid)

v$el.salvador.aid <- gsub("unknown", "2", v$el.salvador.aid)

table(v$religious.groups.in.schools)

v$religious.groups.in.schools <- gsub("unknown", "2", v$religious.groups.in.schools)

table(v$anti.satellite.test.ban)

v$anti.satellite.test.ban <- gsub("unknown", "2", v$anti.satellite.test.ban)

table(v$aid.to.nicaraguan.contras)

v$aid.to.nicaraguan.contras <- gsub("unknown", "2", v$aid.to.nicaraguan.contras)

table(v$mx.missile)

v$mx.missile <- gsub("unknown", "2", v$mx.missile)

table(v$immigration)

v$immigration <- gsub("unknown", "2", v$immigration)

table(v$synfuels.crporation.cutback)

v$synfuels.crporation.cutback <- gsub("unknown", "2", v$synfuels.crporation.cutback)

table(v$education.spending)

v$education.spending <- gsub("unknown", "2", v$education.spending)

table(v$superfund.right.to.sue)

v$superfund.right.to.sue <- gsub("unknown", "2", v$superfund.right.to.sue)

table(v$crime)
v$crime <- gsub("unknown", "2", v$crime)

table(v$duty.free.exports)

v$duty.free.exports <- gsub("unknown", "2", v$duty.free.exports)

table(vo$export.administration.act.south.africa)

v$export.administration.act.south.africa <- gsub("unknown", "2", v$export.administration.act.south.africa)
vo<- v
table(vo$handicapped.infants)
#replace y and n with 1, 2 and 3 qith 2
vo[vo=="2"]<-3
vo[vo=="y"]<-1
vo[vo=="n"]<-2
vo<- data.frame(vo)

View(vo)
vo$ID<- NULL
#making numeric data
vo[] <- lapply(vo, function(x) as.numeric(as.factor(x)))
table(v$class)

vo$class <- gsub("1", "democrat", vo$class, fixed = TRUE)
vo$class <- gsub("2", "republican", vo$class, fixed = TRUE)



#train and split
smp_size <- floor(0.80 * nrow(vo))
set.seed(123)
train_ind <- sample(seq_len(nrow(vo)), size = smp_size)

vtrain <- vo[train_ind, ]
vtest <- vo[-train_ind, ]


######making mulitiple models for SVm with different gamma, cost and kernel
vm1<- svm(as.factor(vtrain$class)~., vtrain)
vm2<- svm(as.factor(vtrain$class)~., vtrain, kernel = "linear")
vm3<- svm(as.factor(vtrain$class)~., vtrain, gamma =0.20 , cost =2)
vm4<- svm(as.factor(vtrain$class)~., vtrain, kernel = "linear", gamma =0.20 , cost =2)
vm5<- svm(as.factor(vtrain$class)~., vtrain, gamma =0.50 , cost =10)
vm6<- svm(as.factor(vtrain$class)~., vtrain, kernel = "linear", gamma =0.50 , cost =10)
vm7<- svm(as.factor(vtrain$class)~., vtrain, gamma =0.0001 , cost =10)
vm8<- svm(as.factor(vtrain$class)~., vtrain, kernel = "linear", gamma =0.0001 , cost =10)
vm9<- svm(as.factor(vtrain$class)~., vtrain, gamma =0.10 , cost =5)
vm10<- svm(as.factor(vtrain$class)~., vtrain, kernel = "linear", gamma =0.10 , cost =5)
vm11<- svm(as.factor(vtrain$class)~., vtrain,scale = FALSE)


###grid search
#obj = tune.svm(vo[3:18],as.factor(vo$class),cost=1:100,gamma= 0.1:10, scale = TRUE)
#obj1= svm(as.factor(vo$class)~., vo, gamma = 0.1, cost = 1, scale = TRUE)
#pp<- predict(obj1, vo)
#table(pp, vo$class)


###prediction
vp1<- predict(vm1, vtest)
vp2<- predict(vm2, vtest)
vp3<- predict(vm3, vtest)
vp4<- predict(vm4, vtest)
vp5<- predict(vm5, vtest)
vp6<- predict(vm6, vtest)
vp7<- predict(vm7, vtest)
vp8<- predict(vm8, vtest)
vp9<- predict(vm9, vtest)
vp10<- predict(vm10, vtest)
vp11<- predict(vm11, vtest)


####confusion matric and accuracy
vt1<- table(vp1, vtest$class)
sum(diag(vt1))/sum(vt1)
vt2<- table(vp2, vtest$class)
sum(diag(vt2))/sum(vt2)

vt3<- table(vp3, vtest$class)
sum(diag(vt3))/sum(vt3)
vt4<- table(vp4, vtest$class)
sum(diag(vt4))/sum(vt4)

vt5<- table(vp5, vtest$class)
sum(diag(vt5))/sum(vt5)
vt6<- table(vp6, vtest$class)
sum(diag(vt6))/sum(vt6)

vt7<- table(vp7, vtest$class)
sum(diag(vt7))/sum(vt7)
vt8<- table(vp8, vtest$class)
sum(diag(vt8))/sum(vt8)

vt9<- table(vp9, vtest$class)
sum(diag(vt9))/sum(vt9)
vt10<- table(vp10, vtest$class)
sum(diag(vt10))/sum(vt10)

vt11<- table(vp11, vtest$class)
sum(diag(vt11))/sum(vt11)



#knn with multiple models
kn1<- knn(vtrain[,2:17], vtest[,2:17], vtrain$class, k =1 )
kn2<- knn(vtrain[,2:17], vtest[,2:17], vtrain$class, k =3 )
kn3<- knn(vtrain[,2:17], vtest[,2:17], vtrain$class, k =5 )
kn4<- knn(vtrain[,2:17], vtest[,2:17], vtrain$class, k =13 )
kn5<- knn(vtrain[,2:17], vtest[,2:17], vtrain$class, k =17 )
kn6<- knn(vtrain[,2:17], vtest[,2:17], vtrain$class, k =19 )
kn7<- knn(vtrain[,2:17], vtest[,2:17], vtrain$class, k =21 )
kn8<- knn(vtrain[,2:17], vtest[,2:17], vtrain$class, k =31 )

#confusion matrix
ktt1<- table(kn1, vtest$class)
sum(diag(ktt1))/sum(ktt1)

ktt2<- table(kn2, vtest$class)
sum(diag(ktt2))/sum(ktt2)
ktt3<- table(kn3, vtest$class)
sum(diag(ktt3))/sum(ktt3)
ktt4<- table(kn4, vtest$class)
sum(diag(ktt4))/sum(ktt4)
ktt5<- table(kn5, vtest$class)
sum(diag(ktt5))/sum(ktt5)
ktt6<- table(kn6, vtest$class)
sum(diag(ktt6))/sum(ktt6)
ktt7<- table(kn7, vtest$class)
sum(diag(ktt7))/sum(ktt7)
ktt8<- table(kn8, vtest$class)
sum(diag(ktt8))/sum(ktt8)

#precison, recall, adm f measure of best models



#for svm
cm = as.matrix(vt3)
precision = diag(cm) / colSums(cm) 
recall = diag(cm) / rowSums(cm) 
f1 = 2 * precision * recall / (precision + recall) 
f1


#for knn
cm1 = as.matrix(ktt4)
precision1 = diag(cm1) / colSums(cm1) 
recall1 = diag(cm1) / rowSums(cm1) 
f11 = 2 * precision1 * recall1 / (precision1 + recall1) 
f11
