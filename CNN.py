#This code is helpful in implementing CNN Model using keras

import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation,Flatten,Conv2D, MaxPooling2D
from keras import initializers,regularizers,optimizers
from sklearn.metrics import confusion_matrix
import pandas as pd
from keras.models import load_model
import numpy as np
from keras import backend as K

npzfile=np.load("mnblur.npz") #change file name
x_train=npzfile['arr_0']
y_train=npzfile['arr_1']
x_test=npzfile['arr_2']
y_test=npzfile['arr_3']

N1=np.shape(x_train)[0]
N=np.shape(x_test)[0]
n=len(np.unique(y_train))

if len(np.shape(x_train))==3:
	x_train=np.reshape(x_train,[np.shape(x_train)[0],np.shape(x_train)[1],np.shape(x_train)[2],1])
	x_test=np.reshape(x_test,[np.shape(x_test)[0],np.shape(x_test)[1],np.shape(x_test)[2],1])

	#change class labels into categorical form
y_train=keras.utils.to_categorical(y_train, num_classes=n)

width=28
height=28
input_shape = (height, width, 1)
nb_filters = 50 # number of convolutional filters to use
pool_size = (2, 2) # size of pooling area for max pooling
kernel_size = (2,2) # convolution kernel size

#Implement layers
model = Sequential()
model.add(Conv2D(nb_filters-20,
						kernel_size,
						input_shape=input_shape,
						activation='relu'))

model.add(Conv2D(nb_filters+10,
						kernel_size,
						activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Conv2D(nb_filters+20,
						kernel_size,
						activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(600, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(n, activation='softmax'))

#Optimizers
Ada=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-10, decay=0.0)
model.compile(optimizer=Ada,loss='categorical_crossentropy',metrics=['accuracy'])

#Model fitting
history=model.fit(x=x_train, y=y_train, batch_size=256,epochs=50,verbose=1)					

#testing
y_pred=model.predict_classes(x_test,batch_size=1000,verbose=0)#testing

conf=confusion_matrix(y_test, y_pred)
accuracy=np.trace(conf)/N