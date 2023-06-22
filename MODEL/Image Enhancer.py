#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import glob
import tensorflow as tf
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import InputLayer,Dense, Conv2D, MaxPool2D, UpSampling2D, Add, Dropout,Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers, Input 
from tensorflow.keras.models import Sequential, Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import pickle
import random
import os
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import PIL
import os
import os.path
from PIL import Image

f = r'C:\Users\win10\ImageEnhancer\Imageset'
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((80,80))
    img.save(f_img)


# In[4]:


plt.figure(figsize=(80,80))
test_folder=r'C:\Users\win10\ImageEnhancer\Imageset'
for i in range(5):
    file = random.choice(os.listdir(test_folder))
    image_path= os.path.join(test_folder, file)
    img=mpimg.imread(image_path)
    ax=plt.subplot(1,5,i+1)
    ax.title.set_text(file)
    plt.imshow(img)


# In[5]:


def create_dataset(test_folder):
   
    img_data_array=[]
    class_name=[]
   
    for dir1 in os.listdir(test_folder):
        for file in os.listdir(os.path.join(test_folder, dir1)):
       
            image_path= os.path.join(test_folder, dir1,  file)
            image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
            image=np.array(image)
            image = image.astype('float32')
            image /= 255 
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name
# extract the image array and class name
img_data, class_name =create_dataset(r'C:\Users\win10\ImageEnhancer')


# In[6]:


with open('img_data','wb') as f:pickle.dump(img_data, f)
print(len(img_data))


# In[7]:


all_images = np.array(img_data)
print(all_images.shape)
#Split test and train data. all_images will be our output images
train_x, val_x = train_test_split(all_images, random_state = 32, test_size=0.2)
print(train_x.shape)
print(val_x.shape)


# In[8]:


#now we will make input images by lowering resolution without changing the size
def pixalate_image(image, scale_percent = 40):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    small_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  
# scale back to original size
    width = int(small_image.shape[1] * 100 / scale_percent)
    height = int(small_image.shape[0] * 100 / scale_percent)
    dim = (width, height)
    low_res_image = cv2.resize(small_image, dim, interpolation =  cv2.INTER_AREA)
    return low_res_image


# In[9]:


train_x_px = []
for i in range(train_x.shape[0]):
    
    print(train_x.shape)

    temp = pixalate_image(train_x[i,:,:,:])
    train_x_px.append(temp)
train_x_px = np.array(train_x_px)   #Distorted images

# get low resolution images for the validation set
val_x_px = []
for i in range(val_x.shape[0]):
    temp = pixalate_image(val_x[i,:,:,:])
    val_x_px.append(temp)
val_x_px = np.array(val_x_px)     #Distorted images


# In[14]:


Input_img = Input(shape=(80, 80, 3))  
    
#encoding architecture
x1 = Conv2D(64, (3, 3), activation='sigmoid', padding='same')(Input_img)
x2 = Conv2D(64, (3, 3), activation='sigmoid', padding='same')(x1)
x3 = MaxPool2D(padding='same')(x2)
x4 = Conv2D(128, (3, 3), activation='sigmoid', padding='same')(x3)
x5 = Conv2D(128, (3, 3), activation='sigmoid', padding='same')(x4)
x6 = MaxPool2D(padding='same')(x5)
encoded = Conv2D(256, (3, 3), activation='sigmoid', padding='same')(x6)
#encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
# decoding architecture
x7 = UpSampling2D()(encoded)
x8 = Conv2D(128, (3, 3), activation='sigmoid', padding='same')(x7)
x9 = Conv2D(128, (3, 3), activation='sigmoid', padding='same')(x8)
x10 = Add()([x5, x9])
x11 = UpSampling2D()(x10)
x12 = Conv2D(64, (3, 3), activation='sigmoid', padding='same')(x11)
x13 = Conv2D(64, (3, 3), activation='sigmoid', padding='same')(x12)
x14 = Add()([x2, x13])
# x3 = UpSampling2D((2, 2))(x3)
# x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x3)
# x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
decoded = Conv2D(3, (3, 3), padding='same',activation='sigmoid', kernel_regularizer=regularizers.l1(10e-10))(x14)
autoencoder = Model(Input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


# In[15]:


autoencoder.summary()


# In[16]:


early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50, verbose=1, mode='min')
model_checkpoint =  ModelCheckpoint('superResolution_checkpoint3.h5', save_best_only = True)


# In[17]:


history = autoencoder.fit(train_x_px,train_x,
            epochs=500,
            validation_data=(val_x_px, val_x),
            callbacks=[early_stopper, model_checkpoint])


# In[18]:


results = autoencoder.evaluate(val_x_px, val_x)
print('val_loss, val_accuracy', results)


# In[19]:


predictions = autoencoder.predict(val_x_px)
n = 4
plt.figure(figsize= (20,10))
for i in range(n):
  ax = plt.subplot(3, n, i+1)
  plt.imshow(val_x_px[i+20])
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax = plt.subplot(3, n, i+1+n)
  plt.imshow(predictions[i+20])
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()


# In[ ]:




