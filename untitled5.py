# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 19:44:05 2017

@author: Bappi Parvez
"""


from __future__ import division, print_function, absolute_import

# Import tflearn and some helpers
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras import backend as K
K.set_image_dim_ordering('tf')

from keras.utils import np_utils
import sys
from skimage import io
from PIL import Image
import glob

PATH = os.getcwd()
# Define data path
data_path = PATH + '/modified-training-dataset'
data_dir_list = os.listdir(data_path)

img_rows=32
img_cols=32
num_channel=3

# Define the number of classes
num_classes = 2

img_data_list=[]


img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
		input_img_resize=cv2.resize(input_img,(32,32))
		img_data_list.append(input_img_resize)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
print (img_data.shape)
num_classes = 2

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:22200]=0
labels[22200:]=1

	  
names = ['human','non-human']
Y = np_utils.to_categorical(labels, num_classes)
x,y = shuffle(img_data,Y, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)



convnet = input_data(shape=[None, 32, 32, 3],data_preprocessing=img_prep,name = 'input_data')
convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = fully_connected(convnet, 512, activation='relu')
convnet = dropout(convnet, 0.5)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
model = tflearn.DNN(convnet, checkpoint_path='faceDetect.tfl.ckpt')

"""
print("start training")
hist = model.fit(X_train, y_train, n_epoch=50, shuffle=True, validation_set=(X_test, y_test),
          show_metric=True, batch_size=96,
          snapshot_epoch=True,
          run_id='faceDetect')

model.save("faceDetect.model")

"""
model.load("faceDetect.model")
print("model loaded")

TP=0
TN=0
FP=0
FN=0


filenames = [img for img in glob.glob("single Check/*.jpg")]

filenames.sort()
images = []
count = 0
total_images = 0
for image in filenames:
    img = scipy.ndimage.imread(image)
    img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')
    test_img = np.array(img)
    #test_img = np.expand_dims(test_img, axis=4)
    outcome = image[11]
    prediction = model.predict([test_img])
    #print(prediction)
    #print(np.array(prediction))
    
    
    
    
    #is_human = np.argmax(prediction[0]) == 1
    #print(outcome)
    if (prediction[0][0]) > (prediction[0][1]):
        if outcome=='n': 
            TP += 1
        else:
            FP +=1
        print("Face Found")
        count +=1
        total_images += 1
    
    else:
        if outcome=='n': 
            FN += 1
        else:
            TN +=1
        print("face not found")
        total_images += 1
        
print("false Positive "+str(FP)+" True Positive "+str(TP)+" True Nagetive "+str(TN)+" False Negative "+str(FN))
print("Total Face Found "+str(count)+" in "+str(total_images)+" images")