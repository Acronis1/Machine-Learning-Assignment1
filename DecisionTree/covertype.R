library(rpart)
library(caret)

setwd("C:/Users/goso/Desktop/házifeladatok/Machine Learning/assignment1")
set.seed(123)
data<-read.csv("covtype.csv",header=FALSE)
toString(c(1:40))
names(data)<- c("Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points","WA1","WA2","WA3","WA4",'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40',  "class")
data['class']

smp_size <- floor(0.8 * nrow(data))
train_ind <- sample(seq_len(nrow(data)), size = smp_size)
train<-data[train_ind,]
test<-data[-train_ind,]

tree_model<-rpart(class~., data=train, method="class",control=rpart.control(minsplit=30, cp=0.001) ) 

prediction<-predict(tree_model, test, type = "class")
confusionMatrix(prediction,test$class)

summary(prediction)
data.frame(table(test$class))

mean(test$class == prediction)
##Accuracy is 0,7278