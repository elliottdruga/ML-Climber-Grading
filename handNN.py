import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from PIL import Image, ImageOps
from numpy import asarray

numImages = 10
num_classes = 8
input_shape = (4000,3000,1)
x = np.zeros((numImages))
y = np.array([1,2,3,4,5])
y = keras.utils.to_categorical(y, num_classes)

for i in range(numImages):
    filename = "H" + i + ".jpg"
    image = Image.open(filename)
    grayimage = ImageOps.grayscale(image)
    imgdata = asarray(grayimage)
    x[0] = imgdata

model = keras.Sequential(
    [
     keras.Input(shape=input_shape),
     layers.Conv2D(32, kernel_size=(3,3), activation="relu"),
     layers.MaxPooling2D(pool_size=(2,2)),
     layers.Conv2D(64, kernel_size=(3,3), activation="relu"),
     layers.MaxPooling2D(pool_size=(2,2)),
     layers.Flatten(),
     layers.Dropout(0.5),
     layers.Dense(num_classes, activation="softmax"),
         
     ]
    )

model.summary()

batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

#score = model.evaluate(x_test, y_test, verbose=0)
#print("Test loss:", score[0])
#print("Test accuracy:", score[1])