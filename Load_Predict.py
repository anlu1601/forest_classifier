#!/usr/bin/env python
# coding: utf-8

# In[24]:


import matplotlib
matplotlib.use('Agg')
import os
from keras.models import load_model
import numpy as np
from PIL import Image
import cv2
#load model.h5 document
model = load_model("inception_1.h5")
model.summary()
#Normalize image size and pixels
def get_inputs(src=[]):
    pre_x = []
    for s in src:
        input = cv2.imread(s)
        input = cv2.resize(input, (224, 224))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        pre_x.append(input)  # input a image
    pre_x = np.array(pre_x) / 255.0
    return pre_x

# images folder address to be classified, get sub_floders for getting all the images address
predict_dir = 'unknown_images'
sub_folders = os.listdir(predict_dir)
print("sub_folders",sub_folders)

#new list saves predicting image address
images_address = []
for sub_folder in sub_folders:
    for item in os.listdir(os.path.join(predict_dir, sub_folder)):
        if item.endswith('.png'):
            item_address = os.path.join(predict_dir, sub_folder, item)
            print(item_address)
            images_address.append(item_address)

#call function, normalize picture, predict, only get array, not sure 0.393313 is birch or spruce?
pre_x = get_inputs(images_address)
pre_y = model.predict(pre_x)
print("Prediction Results:",pre_y)

