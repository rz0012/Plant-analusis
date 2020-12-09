import os
import cv2
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import keras.backend as K
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import efficientnet.keras as efn 
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

efficient_net = efn.EfficientNetB3(
    weights='imagenet',
    input_shape=(64,64,3),
    include_top=False,
    pooling='max'
)


model = Sequential()
model.add(efficient_net)
model.add(Dense(120, activation='relu'))
model.add(Dense(120, activation = 'relu'))
model.add(Dense(3, activation='softmax'))

model.summary()

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

history = model.fit(X, y, batch_size=32, epochs=15, validation_split=0.2, verbose = 1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc,label="accuracy")
plt.legend()
plt.plot(epochs, val_acc, label="val_accuracy")
plt.legend()
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, label="loss")
plt.legend()
plt.plot(epochs, val_loss,label="val_loss")
plt.legend()
plt.title('Training and validation loss')
plt.show()



model_json = model.to_json()
with open("model.json", "w") as json_file :
	json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")

model.save('CNN.model')