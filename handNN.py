import numpy as np
import os
from tensorflow import keras
from keras import layers
from PIL import Image, ImageOps
from numpy import asarray
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#establish number of images, labels, and image resolution
#create label array for training data and convert to one-hot ecoded form
numImages = 17
num_classes = 8
input_shape = (600,800,1)
y = np.array([7,0,6,2,4,4,7,3,3,0,4,6,6,2,2,4,0])
y = keras.utils.to_categorical(y, num_classes)

#import first image and convert to grayscale array
firstImage = Image.open("0.jpg")
firstGrayImage = ImageOps.grayscale(firstImage)
x = asarray(firstGrayImage)

#import remaining images and append to array
for i in range(1,numImages):
    filename = str(i) + ".jpg"
    image = Image.open(filename)
    grayimage = ImageOps.grayscale(image)
    imgdata = asarray(grayimage)
    x = np.dstack([x,imgdata])
  
#convert x value range to 0-1, format tensor for network
x = x.astype("float32") / 255

#duplicate data for augmentation
x = np.dstack([x,x,x,x,x,x,x,x,x,x])
y = np.vstack([y,y,y,y,y,y,y,y,y,y])

#augment data with horizontal and vertical mirroring
for i in range(numImages*10):
    if random.randint(1,4)==2:
        x[i] = np.fliplr(x[i])
    if random.randint(1,4)==2:
        x[i] = np.flipud(x[i])
        

#format data
x = np.expand_dims(x, 2)
x = np.moveaxis(x,3,0)




#create model - two convolution layers followed by max pooling,
#flatten then dense multi-layer percpetron into output
model = keras.Sequential(
    [
     keras.Input(shape=input_shape),
     layers.Conv2D(16, kernel_size=(3,3), activation="relu"),
     layers.MaxPooling2D(pool_size=(2,2)),
     layers.Conv2D(32, kernel_size=(3,3), activation="relu"),
     layers.MaxPooling2D(pool_size=(2,2)),
     layers.Flatten(),
     layers.Dropout(0.5),
     layers.Dense(num_classes, activation="softmax"),
         
     ]
    )

#print model summary
model.summary()

#define parameters for training
batch_size = 4
epochs = 3

#compile and train model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
