import os
import struct
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import sys
print(sys.argv[0])

os.chdir("/run/user/1000/gvfs/sftp:host=192.168.84.212/KALpydev/icr_pickle_files/CNN_Data")

#Emnist data
train_img = 'emnist-byclass-train-images-idx3-ubyte'
train_lbl = 'emnist-byclass-train-labels-idx1-ubyte'
test_img = 'emnist-byclass-test-images-idx3-ubyte'
test_lbl = 'emnist-byclass-test-labels-idx1-ubyte'

with open(train_lbl, 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    train_lbl = np.fromfile(flbl, dtype=np.int8)
        
with open(train_img, 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    train_img = np.fromfile(fimg, dtype=np.uint8).reshape(len(train_lbl), rows, cols)

with open(test_lbl, 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    test_lbl = np.fromfile(flbl, dtype=np.int8)

with open(test_img, 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    test_img = np.fromfile(fimg, dtype=np.uint8).reshape(len(test_lbl), rows, cols)

#Change Background color
	
train_img=np.array(list(map(lambda x:255-np.transpose(x),train_img)))
test_img=np.array(list(map(lambda x:255-np.transpose(x),test_img)))

train_img=train_img[train_lbl>9]
train_lbl=train_lbl[train_lbl>9]

test_img=test_img[test_lbl>9]
test_lbl=test_lbl[test_lbl>9]

train_lbl=train_lbl-10
test_lbl=test_lbl-10

print("Old")
for i in range(5):
    plt.imshow(train_img[i])
    plt.show()



# Removing the noise of background:
    
def fg1(x):
	x[x>np.mean(x)]=1
	return(x)

a=np.array([31,6,35,10,36,41,50])  #Number of images # Manually
b=np.array([60,98,754,132,109,369,347]) #Class Name
x_1 =list()
y_1=list()
x1=list(map(lambda u,v:x_1.extend(test_img[test_lbl==u][0:v]),a,b))
y1=list(map(lambda u,v:y_1.extend(test_lbl[test_lbl==u][0:v]),a,b))

              #31 6 35 10 36 41 50
			#f-60 G-98 j-754 K-115 k-102 p-369 y-347
x_1=np.array(x_1)
y_1=np.array(y_1)

train_img=np.concatenate((train_img,x_1))
train_lbl=np.concatenate((train_lbl,y_1))

size = 3

# generating the kernel
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size

# applying the kernel to the input image

x_train = np.array(list(map(lambda v:cv2.filter2D(v, -1, kernel_motion_blur),train_img)))
x_test = np.array(list(map(lambda v:cv2.filter2D(v, -1, kernel_motion_blur),test_img)))

# x_train=np.array(list(map(lambda x:255-np.transpose(x),x_train)))
print("New")
for i in range(5):
    plt.imshow(x_train[i])
    plt.show()

y_train=train_lbl
y_test=test_lbl

#Normalization
x_train = np.array(list(map(lambda v:(v-v.min())/(v.max()-v.min()),x_train)))
x_test = np.array(list(map(lambda v:(v-v.min())/(v.max()-v.min()),x_test)))
	
p=np.array(pd.Series(np.arange(x_train.shape[0])).sample(x_train.shape[0])) #Reshuffling
x=x_train[p]
y=y_train[p]

#Choose certain numbers of images for making the set balanced

x1=list()
y1=list()

v=2600
for i in range(52):
	x2=x[y==i]
	
	if x2.shape[0]>v:
		y2=np.repeat(i,v)
		y1.extend(y2)
		x2=x2[0:v]
	else:
	
		y2=np.repeat(i,x2.shape[0])
		y1.extend(y2)
		
	x1.extend(x2)
	
	
x1=np.array(x1)
y1=np.array(y1)

#Provide the class number irrespective of whether they are small or capital
y1=np.array(list(map(lambda x:np.int8(x-26) if x>25 else x,y1)))
train_lbl1=np.array(list(map(lambda x:np.int8(x-26) if x>25 else x,train_lbl)))

x1=np.array(list(map(lambda v:fg1(v),x1)))

x_train,x_test,y_train,y_test = train_test_split(x1,y1, test_size = 0.20, stratify=y1)

#Data Augmentation functon

datagen=ImageDataGenerator(
    rotation_range=1,
    width_shift_range=0.10,
    height_shift_range=0.10,
    fill_mode='reflect')
	
def line1(e,c):

	c = cv2.filter2D(c, -1, kernel_motion_blur)
	arr = np.linspace(70,150,48).reshape((24,2))
	u=np.random.shuffle(arr)
	a=np.int(28*e/100)
	c[2:26,a:(a+2)]=arr
	return(c)

#both sides grids function		
def line2(c):

	c = cv2.filter2D(c, -1, kernel_motion_blur)
	arr = np.linspace(70,150,48).reshape((24,2))
	u=np.random.shuffle(arr)
	a=np.int(28*15/100)
	c[2:26,a:(a+2)]=arr
	u=np.random.shuffle(arr)
	a=np.int(28*85/100)
	c[2:26,a:(a+2)]=arr
	return(c)

#Data augmentation and grids in the images		
def data_aug_line(x,y,v):
	x_test2=list()
	y_test2=list()
	
	g=np.array([15,30,45,60,75,85]) #Data Grids Position
	
	for k in range(26):
	
		x1=x[y==k]
		y1=y[y==k]
		x_test1=list()
		y_test1=list()
		i = 0
		n_s=int(v*10/100)
		
		#Data augmentation
		for x_batch,y_batch in datagen.flow(np.reshape(x1,[x1.shape[0],28,28,1]),y1, batch_size=1):
                    i += 1
                    x_batch=x_batch.reshape(28,28)
                    
                    x_test1.append(x_batch)
                    y_test1.extend(y_batch)
                    if np.int(v*50/100)-1:
                        break
                    				
		
		x2=x1.copy()
		x3=list()
		#one grid
		kl= list(map(lambda w:x3.extend(list(map(lambda v:line1(w,v),x2[np.array(pd.Series(np.arange(len(x2))).sample(n_s))]))),g))
		
		#two grids
		kl= x3.extend(list(map(lambda v:line2(v),x2[np.array(pd.Series(np.arange(len(x2))).sample(n_s))])))
		
		x_test1=np.array(x_test1)
		y_test1=np.array(y_test1)
		
		#Add Motion blur noise
		x_test1 = np.array(list(map(lambda v:cv2.filter2D(v, -1, kernel_motion_blur),x_test1)))
		
		x3=np.array(x3)
		y3=np.repeat(k,(len(g)+1)*n_s)
		
		x1=np.concatenate((x_test1,x3))
		y1=np.concatenate((y_test1,y3))
		
		#Normalization
		x1=np.array(list(map(lambda v:(v-v.min())/(v.max()-v.min()),x1)))
		x1=np.array(list(map(lambda v:fg1(v),x1)))
		
		x_test2.extend(x1)
		y_test2.extend(y1)
		
	x_test2=np.array(x_test2)
	y_test2=np.array(y_test2)
	
	return(x_test2,y_test2)	

#Calling function and Creating train-test set
	
x_test2,y_test2=data_aug_line(train_img,train_lbl1,4160)
x_test3,y_test3=data_aug_line(train_img,train_lbl1,1040)


plt.imshow(x_test2[0])
plt.show()
plt.imshow(x_test2[2100])

x_train=np.concatenate((x_train,x_test2))
y_train=np.concatenate((y_train,y_test2))

x_test=np.concatenate((x_test,x_test3))
y_test=np.concatenate((y_test,y_test3))

p=np.array(pd.Series(np.arange(x_train.shape[0])).sample(x_train.shape[0]))

x_train=x_train[p]
y_train=y_train[p]

p=np.array(pd.Series(np.arange(x_test.shape[0])).sample(x_test.shape[0]))

x_test=x_test[p]
y_test=y_test[p]

np.savez("e2.npz",x_train,y_train,x_test,y_test)

