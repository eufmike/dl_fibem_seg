import os, glob
import numpy as np
from skimage.io import imread, imsave, imshow
from PIL import Image, ImageTk
from tqdm.notebook import trange
from core.imageprep import create_crop_idx, crop_to_patch, construct_from_patch


def stack_predict(input_imgpath, 
                  output_imgpath, 
                  cropidx, 
                  model, 
                  rescale = None,
                  patch_size = (256, 256), 
                  predict_threshold = 0.5):
    
    IMG_HEIGHT = patch_size[0]
    IMG_WIDTH = patch_size[1]
                  
    for idx in trange(len(input_imgpath)):
        
        inputimg = input_imgpath[idx]
        
        # load image
        img_tmp = imread(inputimg)
        
        # process rescale
        if rescale is not None:  
            img_tmp = img_tmp * rescale 
        
        # crop the image
        outputimg_tmp = crop_to_patch(img_tmp, cropidx, (IMG_HEIGHT, IMG_WIDTH))
        outputimg_tmp_re = np.reshape(outputimg_tmp, (outputimg_tmp.shape[0], 
                                                      outputimg_tmp.shape[1], 
                                                      outputimg_tmp.shape[2], 1))
        
        # push the crop images into the model
        img_predict_stack = model.predict(outputimg_tmp_re, batch_size = 16, verbose = 1)
        
        outputimg = construct_from_patch(img_predict_stack, 
                                         cropidx, 
                                         target_size = (img_tmp.shape[0], img_tmp.shape[1]))
        
        # threshold the image
        outputimg_T = outputimg > predict_threshold
        
        # save image
        outputimg_T_pillow = Image.fromarray(outputimg_T)
        outputimg_T_pillow.save(os.path.join(output_imgpath, os.path.basename(inputimg)))