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

%load_ext autoreload
%autoreload 2

#%%
# load image
path = "/Volumes/LaCie_DataStorage/PerlmutterData/training/cell_membrane/prepdata"
imgpath = list(paths.list_images(path))
print(imgpath[0])

#%%
# set parameters
seed = 100
batch_size = 16
validation_split = 0.1

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
    os.path.join(path, 'train/images/'),
    class_mode=None,
    color_mode='grayscale',
    batch_size=batch_size,
    subset='training',
    seed=seed)

train_label_generator = label_datagen.flow_from_directory(
    os.path.join(path, 'train/labels'),
    class_mode=None,
    color_mode='grayscale',
    batch_size=batch_size,
    subset='training',
    seed=seed)

valid_image_generator = image_datagen.flow_from_directory(
    os.path.join(path, 'train/images/'),
    class_mode=None,
    color_mode='grayscale',
    batch_size=batch_size,
    subset='validation',
    seed=seed)

valid_label_generator = label_datagen.flow_from_directory(
    os.path.join(path, 'train/labels'),
    class_mode=None,
    color_mode='grayscale',
    batch_size=batch_size,
    subset='validation',
    seed=seed)

#%%
train_generator = zip(train_image_generator, train_label_generator)
valid_generator = zip(valid_image_generator, valid_label_generator)

#%%
from keras.callbacks import ModelCheckpoint

batch_size = 16
checkpointer = ModelCheckpoint('model-test-1.h5', verbose=1, save_best_only=True)

unetmodel = UNet([256, 256])
unetmodel.fit_generator(generator=train_generator, 
                    validation_data = valid_generator,
                    validation_steps = 20,
                    steps_per_epoch = 2000//batch_size,
                    epochs = 3, 
                    verbose=1, 
                    callbacks=[checkpointer]
                    )

