from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from annoy import AnnoyIndex

import scipy.spatial.distance
from scipy.spatial.distance import cosine
from scipy.spatial import distance
from scipy import spatial

from sklearn.cluster import KMeans

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data

#from keras.applications.vgg19 import preprocess_input
# InceptionResNetV2
from keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3

#from keras.applications.inception_resnet_v2 import preprocess_input
#from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2


from tensorflow.keras.layers import Input

import numpy as np

import matplotlib.pyplot as plt

import os
#Load Images from a folder and convert to an array
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        #img = load_img(os.path.join(folder,filename),  target_size=(224, 224)) 
        img = load_img(os.path.join(folder,filename),  target_size=(299, 299)) 
        img = img_to_array(img)
        img = img.reshape((1,) + img.shape)
        img = preprocess_input(img)
        img.flatten()
        if img is not None:
            images.append(img)
    return images

from keras.applications.vgg19 import VGG19
#Cosine similarity: https://medium.com/@salmariazi/computing-image-similarity-with-pre-trained-keras-models-3959d3b94eca
from keras import models, Model

#Create a VGG model and save to specified path
def create_model(save_filepath):
    # loading vgg16 model and using all the layers until the 2 to the    last to use all the learned cnn layers
    #ssl._create_default_https_context = ssl._create_unverified_context
    
    #vgg = VGG19(include_top=True)
    #model2 = Model(vgg.input, vgg.layers[-2].output)
    #model2.save(save_filepath) # saving the model just in case
    
    #input_tensor = Input(shape=(224, 224, 3))
    input_tensor = Input(shape=(299, 299, 3))
    base_model = InceptionV3(input_tensor=input_tensor,weights='imagenet', include_top=True)
    #model2 = Model(base_model.input, base_model.layers[-2].output)
    base_model.save(save_filepath) # saving the model just in case
    
    #input_tensor = Input(shape=(224, 224, 3))
    #input_tensor = Input(shape=(168, 168, 3))
   # base_model = InceptionResNetV2(input_tensor=input_tensor,weights='imagenet', include_top=False) #Include Top can only be true if 299x299
    #model2 = Model(base_model.input, base_model.layers[-2].output)
  #  base_model.save(save_filepath) # saving the model just in case
    return base_model

#Load a model from specified path
def load_model_from_path(save_filepath):
    # loading the model from the saved path
    model = models.load_model(save_filepath)
    #model = load_model(save_filepath)
    return model

#def get_preds(all_imgs_arr,model):
    # getting the extracted features - final shape (number_of_images, 4096)
    #preds = model.predict(all_imgs_arr)
    #return preds

#get predictions (of object?) and return as array
def get_preds(all_imgs_arr):
    #preds_all = np.zeros((len(all_imgs_arr),4096))
    #preds_all = np.zeros((len(all_imgs_arr),1536))
    preds_all = np.zeros((len(all_imgs_arr),1000))
    for j in range(np.array(all_imgs_arr).shape[0]):
        preds_all[j] = model.predict(all_imgs_arr[j])
        
    return preds_all

#Get features
def get_features(all_imgs_arr,model2):
    #model2 = VGG19(weights='imagenet', include_top=False)
    
    featurelist = []
    for i in range(np.array(all_imgs_arr).shape[0]):
        #img_data = img_to_array(all_imgs_arr[i])
        #img_data = np.expand_dims(all_imgs_arr[i], axis=0)
        #img_data = preprocess_input(all_imgs_arr[i])
        img_data = all_imgs_arr[i]
        features = np.array(model2.predict(img_data))
        featurelist.append(features.flatten())
    return featurelist

#K = Numer of images to take into account, Master Image is the vector to check similarity with, preds are predictions
def get_nearest_neighbor_and_similarity(preds, K,MasterImage,saveFile):

    #dims = 4096
    #dims = 25088
    dims = 1000
    #dims = 3
    n_nearest_neighbors = K+1
    trees = 10000
    file_index_to_file_vector = {}
    
    # build ann index (Aproximate Nearest Neighbours)
    t = AnnoyIndex(dims)
    #for i in range(preds.shape[0]):
    i=0
    j=0
    for l in preds:
        
        file_vector = preds[i]
        file_index_to_file_vector[i] = file_vector
        t.add_item(i, file_vector)
        i+=1
    t.build(trees)
    t.save('D:/Users/SCA/'+saveFile)
    
    #for i in range(preds.shape[0]):
    for o in preds:
        master_vector = file_index_to_file_vector[j]
        #Here we assign master vector, SHOULD be one K
    
        named_nearest_neighbors = []
        similarities = []
        nearest_neighbors = t.get_nns_by_item(j, n_nearest_neighbors)
        j+=1

    #Next we print all the neighbours on one axis, should redo new master and nearest for the second axis to plot
    for j in nearest_neighbors:
#         print (j)
        neighbor_vector = preds[j]
    #The distance between objects,/ similarity, cosine for vinkel
        #similarity = 1 - spatial.distance.cosine(master_vector, neighbor_vector)
        similarity = 1 - spatial.distance.cosine(MasterImage, neighbor_vector)
        rounded_similarity = int((similarity * 10000)) / 10000.0
        similarities.append(rounded_similarity)
    return similarities, nearest_neighbors

        
#NOT CURRENTLY WORKING (Supposed to show similar images)        
def get_similar_images(similarities, nearest_neighbors, images1):
    j = 0
    cnt=0
    for i in nearest_neighbors:
        cnt+=1
        #show_img(images1[i])
        #plt.imshow(images1[i])
        #plt.show()
        #img.show(images1[i])
        #print (j)
        #if (similarities[j]<0.8):
        #print (similarities[j])
           # print (j)
        j+=1
    #print (cnt)

