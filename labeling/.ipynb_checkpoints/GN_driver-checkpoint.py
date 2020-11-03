#!/usr/bin/env python
# coding: utf-8

# In[4]:


import labeling.Divide_DataSet as dv
from labeling.GN import create_model
from labeling.GN import train_model
from labeling.GN import predict
import os


def divideDataset(original_folder, output_folder):
    dv.divide(original_folder, output_folder)
    train_root = output_folder + '/train/'
    validation_root = output_folder + '/val/'
    test_root= output_folder + '/test/'
    return train_root, validation_root, test_root
    
def check_model(folder, model, train_root, validation_root, test_root):
    if os.path.exists(model):
        print("It exists. Predicting...")
        name, results = predict(folder, model)
#         print("images_names",name)
#         print("predictions",results)
    else:
        print("Model doesn't exist. Creating a new one...")
        model = create_model()
        trained_model = train_model(model, train_root, validation_root, test_root)
        name, results = predict(folder, trained_model)
    return name, results


# In[5]:


if __name__ == "__main__":
    model = 'inception_1.h5'
    to_be_predict_folder = 'unknown_images'
     # step1: create trainset, validationset, testingset
    (train_root, validation_root, test_root) = divideDataset("dataset",  "after_divide_dataset") 
    # step2: build and predict
    (name, results) = check_model(to_be_predict_folder, model, train_root, validation_root, test_root)


# In[ ]:




