#data import and handling is based on the tutorial from github!

import os
import glob
import keras
import numpy as np

from PIL import Image
from theano import config

from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization

np.random.seed(1)

path = "TrainImages"
files = glob.glob(os.path.join(path, '*.pgm'))
print("found ", len(files), " files")


images = []
image_names = []

for filename in files:
	image_names.append(os.path.basename(filename))
	with Image.open(filename) as img:
		images.append(np.array(img))

print(image_names[1])
Image.fromarray(images[1]).save("temp.png")


#merging the images into one big array:

img_array = np.array(images, dtype=config.floatX)
print("shape si : " ,img_array.shape)


#let's create the labels:
classes = []
for name in image_names:
    if 'neg' in name:
        classes.append(0)
    else:
        classes.append(1)

print(len(classes))  #length is 1050 good
print(sum(classes))  #from which positives are 550, class assignment was successful


#normalization:
mean = img_array.mean()
stddev = img_array.std()
img_array = (img_array - mean) / stddev

#doing the same for test data:

path = 'TestImages'
files = glob.glob(os.path.join(path, '*.pgm'))
print("test: ", len(files))

#from image_preprocessing import resize_and_crop

test_images = []

for filename in files:
    with Image.open(filename) as img:
        img_resized = img.resize((100,40)) #the original size used for training
        test_images.append(np.array(img_resized))

Image.fromarray(test_images[2]).save("temp3.png") #just to see if it works

test_images = np.array(test_images)

#normalizing
mean = test_images.mean()
stddev = test_images.std()
test_images = (test_images - mean) / stddev

#flattening
test_images_flat = test_images.reshape(test_images.shape[0],-1)

#and at last, the classes. test dataset only conatins cars:
test_classes = [1] * len(files)

#building the CNN model, but first some inistial settings regrading the colour scale:

n_channels = 1 # for grey-scale, 3 for RGB, but usually already present in the data

if keras.backend.image_dim_ordering() == 'th':
    # Theano ordering (~/.keras/keras.json: "image_dim_ordering": "th")
    train_img = img_array.reshape(img_array.shape[0], n_channels, img_array.shape[1], img_array.shape[2])
    test_img = test_images.reshape(test_images.shape[0], n_channels, test_images.shape[1], test_images.shape[2])
else:
    # Tensorflow ordering (~/.keras/keras.json: "image_dim_ordering": "tf")
    train_img = img_array.reshape(img_array.shape[0], img_array.shape[1], img_array.shape[2], n_channels)
    test_img = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], n_channels)


input_shape = train_img.shape[1:]
print("input shape: ", input_shape)

def createMyModel():
    
    model = Sequential()

    n_filters = 16
    # this applies n_filters convolution filters of size 5x5 resp. 3x3 each in the 2 layers below

    # Layer 1
    model.add(Convolution2D(n_filters, 3, 3, border_mode='valid', input_shape=input_shape))    
    # input shape: 100x100 images with 3 channels -> input_shape should be (3, 100, 100) 
    model.add(BatchNormalization())
    model.add(Activation('relu'))  # ReLu activation
    model.add(MaxPooling2D(pool_size=(2, 2))) # reducing image resolution by half
    model.add(Dropout(0.3))  # random "deletion" of %-portion of units in each batch

    # Layer 2
    model.add(Convolution2D(n_filters, 3, 3))  # input_shape is only needed in 1st layer
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten()) # Note: Keras does automatic shape inference.
    
    # Full Layer
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(1,activation='sigmoid'))
    
    return model


model = createMyModel()

loss = 'binary_crossentropy' 
optimizer = 'sgd' 

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

epochs = 15

#lets jump into validation already:


#uncomment to see:
"""

history = model.fit(train_img, classes, batch_size=32, nb_epoch=epochs, validation_split=0.15)

"""


#with 15% validation from the training set the accuracy is 0,981

#using test data for validation:
validation_data = (test_img, test_classes)



#uncomment to see:

"""

history = model.fit(train_img, classes, batch_size=32, nb_epoch=epochs, validation_data=validation_data)

"""



#validation accuracy is really low 3%

#As mentioned in the notebook from github, generating new samples might help the network to generalize better
#let's do that here too :

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

classes_array = np.array(classes)
np.random.seed(0)

model = createMyModel()
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

# fits the model on batches with real-time data augmentation:
history = model.fit_generator(datagen.flow(train_img, classes_array, batch_size=16),
                    samples_per_epoch=len(train_img), nb_epoch=epochs,
                    validation_data=validation_data)


test_pred = model.predict_classes(test_img)
acc = accuracy_score(test_classes, test_pred)
print("our final accuracy with generated samples: ", acc)
#accuracy is 0,7235, pretty good