import os
import sys
import pandas as pd
import numpy as np
import graphviz
import io
import matplotlib
import pydotplus

from scipy import misc
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image
from sklearn.ensemble import RandomForestClassifier



data_congress=pd.read_csv('Congressional_train.csv',na_values = ['unknown', '.'])
x_test=pd.read_csv('Congressional_test.csv',na_values = ['unknown', '.'])

data_rep=data_congress.loc[data_congress['class'] == 'republican']
data_dem=data_congress.loc[data_congress['class'] == 'democrat']



data_rep = data_rep.fillna(data_rep.mode().iloc[0])
data_dem = data_dem.fillna(data_dem.mode().iloc[0])

print("valami")


data_congressfilled=pd.concat([data_rep,data_dem])


print(data_congressfilled)


data_congressfilled2 = data_congress.fillna(data_congress.mode().iloc[0])
x_testfilled = x_test.fillna(x_test.mode().iloc[0])

for column in data_congressfilled.columns:
    if data_congressfilled[column].dtype == type(object):
        le = LabelEncoder()
        data_congressfilled[column] = le.fit_transform(data_congressfilled[column])

for column in x_testfilled.columns:
    if x_testfilled[column].dtype == type(object):
        le = LabelEncoder()
        x_testfilled[column] = le.fit_transform(x_testfilled[column])

#218 row 18 attr
#let's build two models on filled and regular

features=list(data_congressfilled)[2:] #all attributes but class
target=list(data_congressfilled)[1:2] #only class

x_train=data_congressfilled[features]
y_train=data_congressfilled[target]

x_test1=x_test[features]
x_test2=x_testfilled[features]
print(x_test1,x_test2)

#let's train the model

c=DecisionTreeClassifier(min_samples_leaf=5,criterion="gini",class_weight='balanced')

dt=c.fit(x_train,y_train)
"""
#making a pic of the tree:
dotfile = open("dtree2.dot", 'w')
tree.export_graphviz(dt, out_file = dotfile, feature_names =features)



dot_data = StringIO()

export_graphviz(dt, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

def show_tree(tree,features,path):
	f=io.StringIO()
	export_graphviz(tree,out_file=f, feature_names=features)
	pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
	img=misc.imread(path)
	plt.rcParams["figure.figsize"]=(20,20)
	plt.imshow(img)

show_tree(dt,features,'dec_tree_1')"""

#predicting:

for column in x_test2.columns:
    if x_test2[column].dtype == type(object):
        le = LabelEncoder()
        x_test2[column] = le.fit_transform(x_test2[column])



y_pred = c.predict(x_test2)
print("asd")

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(y_pred)
    


print("vmi")
print(list(x_test2)[:1])


y_pred2=y_pred.astype(str)
y_pred2[y_pred==1]='republican'
y_pred2[y_pred==0]='democrat'

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(y_pred)
    print(y_pred2)


print("asdasd")



solution=np.vstack((x_test['ID'],y_pred2))
solution=np.column_stack((x_test['ID'],y_pred2))
print(solution)
np.savetxt('dectree_precise.csv', solution, fmt='%s', delimiter=',', header='ID,class')

#only 4 changed! between simplified and full dtree


#Let's do random forests, it is worrse than the tree so outcommented:


