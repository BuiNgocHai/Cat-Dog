#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, shutil, random, glob
import bcolz
import keras
import keras.preprocessing.image
from keras.layers import Input, Flatten, Dense, Dropout, Activation, BatchNormalization, GlobalMaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import VGG19
from keras.models import Model
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import numpy as np
import scipy


# 1.Change directory data

# In[2]:


TRAIN_DIR = '/home/hai/Desktop/Cat&Dog/all/train/'
f = '/home/hai/Desktop/Cat&Dog/train/'
for img in os.listdir(TRAIN_DIR):
    dogs_or_cats = 'dogs' if 'dog' in img else 'cats'
    shutil.copy(TRAIN_DIR+img, f'train/{dogs_or_cats}/{img}')


# Generate Data

# In[3]:


gen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_data = gen.flow_from_directory('train', target_size=(224, 224), batch_size=1, shuffle=False)


# In[4]:


train_filenames = train_data.filenames
bcolz.carray(train_filenames, rootdir='train_filenames', mode='w').flush()
train_y = keras.utils.to_categorical(train_data.classes)
bcolz.carray(train_y, rootdir='train_y', mode='w').flush()


# In[5]:


base_model = VGG19(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3),
    pooling=None
)


# In[11]:


train_X = base_model.predict_generator(train_data, steps=train_data.n)
bcolz.carray(train_X, rootdir='train_X', mode='w').flush()


# In[12]:


trn_ids = np.random.randint(25000, size=20000)
val_ids = np.delete(np.arange(25000), trn_ids)

trn_X = train_X[trn_ids, ...]
trn_y = train_y[trn_ids]

val_X = train_X[val_ids, ...]
val_y = train_y[val_ids]


# In[13]:


inputs = Input(shape=(7, 7, 512))
# x = keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(inputs)
# x = Flatten()(x)
# x = Dense(4096)(x)

x = GlobalMaxPooling2D()(inputs)
x = Dense(4096)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(2)(x)
x = BatchNormalization()(x)
predictions = Activation('softmax')(x)

model = Model(inputs, predictions)


# In[15]:


model.summary()


# In[16]:


model.compile(Adam(lr=1e-4), 'categorical_crossentropy', metrics=['accuracy'])


# In[18]:


model.fit(x=trn_X, y=trn_y, batch_size=20000, epochs=40, validation_data=(val_X, val_y), verbose=2)


# In[19]:


model.save('VGG19_model.h5')


# In[20]:


from keras.models import load_model
model = load_model('VGG19_model.h5')


# In[28]:


import cv2
TEST_DIR = '/home/hai/Desktop/Cat&Dog/all/test'
def process_test_data():
    testing_data = []
    for img in os.listdir(TEST_DIR):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path)
        img = cv2.resize(img, (224,224))
        testing_data.append([np.array(img), img_num])
    #shuffle(testing_data)
    #np.save('test_data.npy', testing_data)
    return testing_data


# In[30]:


test_data = process_test_data()
with open('submission_file.csv','w') as f:
    f.write('id,label\n')
            
with open('submission_file.csv','a') as f:
    for data in test_data:
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(-1,7,7,512)
        model_out = model.predict([data])[0]
        f.write('{},{}\n'.format(img_num,model_out[1]))

