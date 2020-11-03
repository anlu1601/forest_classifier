#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.layers import Input, concatenate
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model,np_utils
from keras import regularizers
import keras.metrics as metric
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import numpy as np
import pandas as pd
import os
import cv2

# Global Constants
LRN2D_NORM = True #default
EPOCH = 5
batch_size = 32
IM_WIDTH = 224 
IM_HEIGHT = 224
WEIGHT_DECAY = 0.0005
DATA_FORMAT = 'channels_last' # Theano:'channels_first' Tensorflow:'channels_last'
DROPOUT = 0.4
NB_CLASS = 2
    


# In[2]:


def conv2D_lrn2d(x, filters, kernel_size, strides=(1,1), padding='same', 
                 data_format=DATA_FORMAT, dilation_rate=(1,1) ,activation='relu',
                 use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zero',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, lrn2d_norm=LRN2D_NORM, 
                 weight_decay=WEIGHT_DECAY):   
    #l2 normalization
    if weight_decay:
        kernel_regularizer = regularizers.l2(weight_decay)
        bias_regularizer = regularizers.l2(weight_decay)
    else:
        kernel_regularizer = None
        bias_regularizer = None
        X = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, 
                 data_format=data_format, dilation_rate = dilation_rate, activation=activation,
                 use_bias=use_bias, kernel_initializer=kernel_initializer, 
                 bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                 bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,
                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                 )(x)

    if lrn2d_norm:
        #batch normalization
        x=BatchNormalization()(x)
    return x


# In[3]:


def inception_module(x,params,concat_axis,padding='same',data_format=DATA_FORMAT,
                     dilation_rate=(1,1),activation='relu',use_bias=True,
                     kernel_initializer='glorot_uniform',bias_initializer='zeros',
                     kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,
                     kernel_constraint=None,bias_constraint=None,lrn2d_norm=LRN2D_NORM,weight_decay=None):
    (branch1,branch2,branch3,branch4)=params
    if weight_decay:
        kernel_regularizer=regularizers.l2(weight_decay)
        bias_regularizer=regularizers.l2(weight_decay)
    else:
        kernel_regularizer=None
        bias_regularizer=None
    #1x1
    pathway1=Conv2D(filters=branch1[0],kernel_size=(1,1),strides=1,padding=padding,
                    data_format=data_format,dilation_rate=dilation_rate,activation=activation,
                    use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint)(x)

    #1x1->3x3
    pathway2=Conv2D(filters=branch2[0],kernel_size=(1,1),strides=1,padding=padding,
                    data_format=data_format,dilation_rate=dilation_rate,activation=activation,
                    use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint)(x)
    pathway2=Conv2D(filters=branch2[1],kernel_size=(3,3),strides=1,padding=padding,
                    data_format=data_format,dilation_rate=dilation_rate,activation=activation,
                    use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint)(pathway2)

    #1x1->5x5
    pathway3=Conv2D(filters=branch3[0],kernel_size=(1,1),strides=1,padding=padding,
                    data_format=data_format,dilation_rate=dilation_rate,activation=activation,
                    use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint)(x)
    pathway3=Conv2D(filters=branch3[1],kernel_size=(5,5),strides=1,padding=padding,
                    data_format=data_format,dilation_rate=dilation_rate,activation=activation,
                    use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint)(pathway3)

    #3x3->1x1
    pathway4=MaxPooling2D(pool_size=(3,3),strides=1,padding=padding,data_format=DATA_FORMAT)(x)
    pathway4=Conv2D(filters=branch4[0],kernel_size=(1,1),strides=1,padding=padding,
                    data_format=data_format,dilation_rate=dilation_rate,activation=activation,
                    use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint)(pathway4)

    return concatenate([pathway1,pathway2,pathway3,pathway4],axis=concat_axis)


# In[4]:


def build_model():      
    if DATA_FORMAT=='channels_first':
        INP_SHAPE=(3,224,224)
        img_input=Input(shape=INP_SHAPE)
        CONCAT_AXIS=1
    elif DATA_FORMAT=='channels_last':
        INP_SHAPE=(224,224,3)
        img_input=Input(shape=INP_SHAPE)
        CONCAT_AXIS=3
    else:
        raise Exception('Invalid Dim Ordering')

    x=conv2D_lrn2d(img_input,64,(7,7),2,padding='same',lrn2d_norm=False)
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)
    x=BatchNormalization()(x)

    x=conv2D_lrn2d(x,64,(1,1),1,padding='same',lrn2d_norm=False)

    x=conv2D_lrn2d(x,192,(3,3),1,padding='same',lrn2d_norm=True)
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)

    x=inception_module(x,params=[(64,),(96,128),(16,32),(32,)],concat_axis=CONCAT_AXIS) #3a
    x=inception_module(x,params=[(128,),(128,192),(32,96),(64,)],concat_axis=CONCAT_AXIS) #3b
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)

    x=inception_module(x,params=[(192,),(96,208),(16,48),(64,)],concat_axis=CONCAT_AXIS) #4a
    x=inception_module(x,params=[(160,),(112,224),(24,64),(64,)],concat_axis=CONCAT_AXIS) #4b
    x=inception_module(x,params=[(128,),(128,256),(24,64),(64,)],concat_axis=CONCAT_AXIS) #4c
    x=inception_module(x,params=[(112,),(144,288),(32,64),(64,)],concat_axis=CONCAT_AXIS) #4d
    x=inception_module(x,params=[(256,),(160,320),(32,128),(128,)],concat_axis=CONCAT_AXIS) #4e
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)

    x=inception_module(x,params=[(256,),(160,320),(32,128),(128,)],concat_axis=CONCAT_AXIS) #5a
    x=inception_module(x,params=[(384,),(192,384),(48,128),(128,)],concat_axis=CONCAT_AXIS) #5b
    x=AveragePooling2D(pool_size=(7,7),strides=1,padding='valid',data_format=DATA_FORMAT)(x)

    x=Flatten()(x)
    x=Dropout(DROPOUT)(x)
    x=Dense(NB_CLASS,activation='linear')(x)
    x=Dense(NB_CLASS,activation='softmax')(x)
    # x=Dense(output_dim=NB_CLASS,activation='linear')(x)
    # x=Dense(output_dim=NB_CLASS,activation='softmax')(x)

    return x,img_input,CONCAT_AXIS,INP_SHAPE,DATA_FORMAT


# In[5]:


def create_model():
    x,img_input,CONCAT_AXIS,INP_SHAPE,DATA_FORMAT=build_model()
    model=Model(img_input,[x])
    #model.summary()

    # Save a PNG of the Model Build
    plot_model(model,to_file='GoogLeNet.png')

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc',metric.top_k_categorical_accuracy])
    print('Model Compiled')
    return model


# In[6]:


def train_model(_model, train_root, validation_root, test_root):
    # train data
    train_datagen = ImageDataGenerator(
        rotation_range = 30,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        featurewise_center = False
        # featurewise_center: 3 channels of the original image value-the mean value of the 3 channels of the original image value
    )
    train_generator = train_datagen.flow_from_directory(
        train_root,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size = batch_size,
    )

    # valid data
    valid_datagen = ImageDataGenerator(
        rotation_range = 30,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        featurewise_center = False
    )
    valid_generator = train_datagen.flow_from_directory(
        validation_root,
        target_size = (IM_WIDTH, IM_HEIGHT),
        batch_size = batch_size,
    )

    #test data
    test_datagen = ImageDataGenerator(
        rotation_range = 30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip=True,
        featurewise_center=False
    )
    test_generator = train_datagen.flow_from_directory(
        test_root,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size = batch_size,
    )

    _model.fit(train_generator,validation_data=valid_generator,epochs=EPOCH,
                        steps_per_epoch=train_generator.n/batch_size,
                       validation_steps=valid_generator.n/batch_size)
#     _model.save('inception_2.h5')
    _model.save('./labeling/inception_1.h5')
    _model.METRICS=['acc',metric.top_k_categorical_accuracy]
    loss,acc,top_acc=_model.evaluate(test_generator,steps=test_generator.n/batch_size)
    print('Test result:loss:%f,acc:%f,top_acc:%f'%(loss,acc,top_acc))
    return _model


# In[7]:


def predict(_folder,_model):
    #load model.h5 document
    model = load_model('./labeling/inception_1.h5')
    #model.summary()

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
    predict_dir = _folder
    sub_folders = os.listdir(predict_dir)
    print("sub_folders",sub_folders)
    classes = os.listdir('./labeling/input')

    outfile = []

    #new list saves predicting image address
    images_address = []
    images_names = []
    for sub_folder in sub_folders:
        for item in os.listdir(os.path.join(predict_dir, sub_folder)):
            if item.endswith('.png'):
                item_address = os.path.join(predict_dir, sub_folder, item)
                #print(item_address)
                images_address.append(item_address)
                images_names.append(item)

    #call function, normalize picture, predict, only get array, not sure 0.393313 is birch or spruce?
    pre_x = get_inputs(images_address)
    pre_y = model.predict(pre_x)
    #y_classes = pre_y.argmax(axis=-1)
    #print("Num--Images_Names----Predictions------Predicted species")
    for i in range(len(images_names)):
        class_index = np.argmax(pre_y[i])
        #print("{}--{}----{}-----{}".format(i, images_names[i], pre_y[i], classes[class_index]))
        percentage = "{:.2%}".format(pre_y[i][class_index])
        outfile.append((classes[np.argmax(pre_y[i])], percentage, images_address[i]))

    column_names = ['Prediction', 'Accuracy', 'Image']
    # print(outfile[0][1])
    out = np.asarray(outfile)
    # print(out)
    df = pd.DataFrame(out, columns=column_names)
    # print(df)
    df.to_csv('prediction_list_Inception.csv', index=False)
    print("Prediction list created")

    return images_names, pre_y


# In[ ]:




