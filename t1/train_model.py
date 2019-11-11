#%%
# import os, glob, shutil
import os, sys
import cv2
import numpy as np
import uuid
import tensorflow as tf
from skimage.io import imread, imsave, imshow
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from core.imageprep import random_crop, crop_generator, random_crop_batch
from core.models import UNet
from imutils import paths
import itertools

#%load_ext autoreload
#%autoreload 2

#%%
#load image
#path = "/Volumes/LaCie_DataStorage/PerlmutterData/"
path = "D:/PerlmutterData/"
imgdir = "training/cell_membrane/prepdata"
imgpath = os.path.join(path, imgdir)

imgpath_all = list(paths.list_images(path))
print(imgpath_all[0])

#%%
# set parameters
seed = 100
batch_size = 16
validation_split = 0.1
training_sample_size = len(imgpath_all)
IMG_HEIGHT = 256
IMG_WIDTH = 256

#%% 
# create argments for data generator
data_gen_args = dict(
                featurewise_center=True,
                featurewise_std_normalization=True,
                horizontal_flip=True,
                vertical_flip=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.07,
                zoom_range=0.2,
                validation_split=validation_split, 
                # fill_mode='constant',
                # cval=0.,
                # rescale=1.0/255.0,
                )

#%%
image_datagen = ImageDataGenerator(**data_gen_args)
label_datagen = ImageDataGenerator(**data_gen_args)

#%%
train_image_generator = image_datagen.flow_from_directory(
    os.path.join(imgpath, 'train/images/'),
    class_mode=None,
    color_mode='grayscale',
    batch_size=batch_size,
    subset='training',
    seed=seed)

train_label_generator = label_datagen.flow_from_directory(
    os.path.join(imgpath, 'train/labels'),
    class_mode=None,
    color_mode='grayscale',
    batch_size=batch_size,
    subset='training',
    seed=seed)

valid_image_generator = image_datagen.flow_from_directory(
    os.path.join(imgpath, 'train/images/'),
    class_mode=None,
    color_mode='grayscale',
    batch_size=batch_size,
    subset='validation',
    seed=seed)

valid_label_generator = label_datagen.flow_from_directory(
    os.path.join(imgpath, 'train/labels'),
    class_mode=None,
    color_mode='grayscale',
    batch_size=batch_size,
    subset='validation',
    seed=seed)

#%%
# merge image and label generator
train_generator = zip(train_image_generator, train_label_generator)
valid_generator = zip(valid_image_generator, valid_label_generator)

#%%
# create folder for saving the model


#%%
# training
from keras.callbacks import ModelCheckpoint
from datetime import datetime

print("Start training")

# checkpointer
# check folder
if not 'model' in os.listdir(path):
    os.mkdir(os.path.join(path, 'model'))
# set up the checkpointer

# model
checkpointer = ModelCheckpoint('model_' + datetime.now().strftime("%Y_%m_%d_%H_%M") + '.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# calculate steps_per_epoch
steps_per_epoch = training_sample_size * (1-validation_split) // batch_size
print("Steps per epoch: {}".format(steps_per_epoch))

#%%
# prepare the model
unetmodel = UNet([IMG_HEIGHT, IMG_WIDTH])

# train the model
unetmodel.fit_generator(generator=train_generator, 
                    validation_data = valid_generator,
                    validation_steps = 20,
                    steps_per_epoch = steps_per_epoch,
                    epochs = 500, 
                    verbose=1, 
                    callbacks=[checkpointer]
                    )
