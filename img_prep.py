#%%
import os, glob, shutil
import cv2
import numpy as np
from skimage.io import imread, imsave, imshow
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from core.imageprep import random_crop, crop_generator, random_crop_batch

# %load_ext autoreload
# %autoreload 2

#%%
# set parameters
img_crop_shape_min = 256
img_crop_shape_max = 256

#%%
# Prepare the training dataset
# Specify the data folder
mainpath = '/Volumes/LaCie_DataStorage/PerlmutterData/training'
labeltype = 'cell_membrane'
data_path = os.path.join(mainpath, labeltype)

# %%
# create list for filenames
imglist = glob.glob(os.path.join(data_path, 'data', 'images', '*', '*.tif'), recursive=True)
labellist = glob.glob(os.path.join(data_path, 'data', 'labels', '*', '*.tif'), recursive=True)

#%%
# Create output folder
if not 'prepdata' in os.listdir(data_path):
    foldernames = ['train',]
    os.mkdir(os.path.join(data_path, 'prepdata'))
    for foldername in foldernames:    
        os.mkdir(os.path.join(data_path, 'prepdata', foldername))
        foldernames_class = ['images', 'labels']
        for foldername_class in foldernames_class:
            os.mkdir(os.path.join(data_path, 'prepdata', foldername, foldername_class))
            label_names = ['cell_membrane']
            for label_name in label_names: 
                os.mkdir(os.path.join(data_path, 'prepdata', 
                                        foldername, foldername_class, label_name))  

#%%
# Batch Random Crop
ipfolder = os.path.join(data_path, 'data')
# create train dataset 
opfolder = os.path.join(data_path, 'prepdata', 'train')
random_crop_batch(ipfolder, opfolder, 'cell_membrane', [img_crop_shape_min, img_crop_shape_max], 20, 100)

