import os
import glob
import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2 						#pip install opencv-python
import tensorflow as tf 
import itertools
import datetime

from scipy.stats import mode
from numpy import loadtxt

from PIL import Image
from theano import config

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization


DIM = 100   #define what resolution the pics should be, DIM X DIM
EPOCHS = 100
OPT = 'adam'
LOSS = 'sparse_categorical_crossentropy'

np.random.seed(1)

#import the fruit names into an array:
fruit_names = []

fruit_names = loadtxt("fruits.txt", delimiter=";", unpack=False, dtype='str')

path = fruit_names[0]
files = glob.glob(os.path.join(path, '*.jpg'))
print("found ", len(files), " files")

images = []
classes = []

for fruit in fruit_names:
	
	path = fruit
	files = glob.glob(os.path.join(path, '*.jpg'))
	
	for filename in files:
		classes.append(fruit)
		with Image.open(filename) as img:
			images.append(np.array(img.resize((DIM,DIM))))

#encoding the string labels:
le = preprocessing.LabelEncoder()
le.fit(fruit_names)
#transformation will be done later
 
#To see what the normalization does:
plt.imshow(images[1], cmap = plt.cm.binary)
plt.show()
test = keras.utils.normalize(images[1], axis=2)
plt.imshow(test, cmap = plt.cm.binary)
plt.show()

i=0



for j in range(0,len(images)):
	try:
		images[j] = keras.utils.normalize(images[j], axis=2)
		#images[j] = keras.utils.normalize(images[j], axis=1)
		print(i)
	except:
		print("it seems that this image is not in RGB")

	i+=1
		

to_delete = []


for i in range(0,len(images)):
	if images[i].shape[2] != 3:
		to_delete.append(i)
	if i == 126:
		print(images[i].shape)

for i in reversed(to_delete):
	images.pop(i)
	classes.pop(i)
classes = le.transform(classes) #here is the label transoformation, which return an ndarray, which does not have .pop method

img_array = np.array(images, dtype=config.floatX)

x_train, x_test, y_train, y_test = train_test_split(img_array, classes, test_size=0.2, random_state=42)

print("shape of the training images: ",x_train.shape)
print("shape of the labels: ",y_train.shape)

n_filters = 16

"""
#######################################################
model = Sequential()
model.add(Flatten(input_shape=(DIM,DIM,3)))
model.add(Dense(150, activation=tf.nn.relu))
model.add(Dense(150, activation=tf.nn.relu))
model.add(Dense(30, activation=tf.nn.softmax)) #30 is the number of classes that we have

model.compile(optimizer=OPT ,loss=LOSS ,metrics=['accuracy'])

currentDT = datetime.datetime.now()

model.fit(x_train, y_train, epochs=EPOCHS)

print("Building the DNN model took: ", datetime.datetime.now()-currentDT," with ",EPOCHS," number of EPOCHS")


val_loss, val_acc = model.evaluate(x_test, y_test)
print("the accuracy of the model is: ", val_acc)

predictions = model.predict_classes(x_test)
print(predictions)


print("size of pred is : ", len(predictions))
predictions = list(le.inverse_transform(predictions.astype(int)))
y_test = list(le.inverse_transform(y_test.astype(int)))

print("the predictions are: ",predictions)
print("the actual labels: ",y_test)
#######################################################
"""

#with 150x150 accuracy is 5%
#with 200x200 accuracy is 6,28%
#with 80x80 accuracy is 25,1%
#with 75x75 accuracy is 27,7% without normalization 3%
#with 60x60 accuracy is 24,3%
#with 50x50 accuracy is 22%

##########################
#CNN based on the example of TensorFlow: 

#######################################################

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

print("Creating CNN model")

LR = 1e-3

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(DIM, DIM, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))

model.add(Conv2D(128, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size=(7, 7), activation='relu'))
model.add(Dropout(0.35))


model.add(Flatten())

model.add(Dense(120, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(30, activation='softmax'))

model.compile(loss=LOSS, optimizer=OPT, metrics=['accuracy'])
#fitting:

currentDT = datetime.datetime.now()

model.fit(x_train, y_train, epochs=EPOCHS)

print("Building the DNN model took: ", datetime.datetime.now()-currentDT," with ",EPOCHS," number of EPOCHS")

val_loss, val_acc = model.evaluate(x_test, y_test)
print("the accuracy of the model is: ", val_acc)

predictions = model.predict_classes(x_test)
print(predictions)


print("size of pred is : ", len(predictions))
predictions = list(le.inverse_transform(predictions.astype(int)))
y_test = list(le.inverse_transform(y_test.astype(int)))

print("the predictions are: ",predictions)
print("the actual labels: ",y_test)

#######################################################

#function for the plot of conf matrix:
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

cnf_matrix = confusion_matrix(y_test, predictions)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=fruit_names,
                      title=str('Confusion matrix, acc='+str(val_acc)))
plt.show()



#######################################################
"""
Uncomment this to see how the model performs with image generator
results so far with dim = 75, epoch = 25 acc=23,4%


from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),
                    samples_per_epoch=len(x_train), nb_epoch=EPOCHS)

val_loss, val_acc = model.evaluate(x_test, y_test)
print("the accuracy of the model is: ", val_acc)

predictions = model.predict_classes(x_test)
print("the predictions are: ",predictions)
print("the actual labels: ",y_test)


print("size of pred is : ", len(predictions))
predictions = list(le.inverse_transform(predictions.astype(int)))
y_test = list(le.inverse_transform(y_test.astype(int)))

cnf_matrix = confusion_matrix(y_test, predictions)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=fruit_names,
                      title=str('Confusion matrix, acc='+str(val_acc)))
plt.show()


"""
#######################################################