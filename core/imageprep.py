import os, glob
import numpy as np
import math
import uuid
from skimage.io import imread, imsave, imshow
from PIL import Image, ImageTk
from typing import Union, Any, List, Tuple
from tqdm.notebook import trange
import matplotlib.pyplot as plt

def dir_checker(folder_name, path):
    if not folder_name in os.listdir(path):
        print('{} does not exist in {}'.format(folder_name, path))
        os.mkdir(os.path.join(path, folder_name))
    else: 
        print('{} exists in {}'.format(folder_name, path))

def random_crop(imgs, random_range, seed=None):
    # Note: image_data_format is 'channel_last'
    # assert img.shape[2] == 3
    height, width = imgs[0].shape
    
    range_low = random_range[0]
    range_high = random_range[1] + 1

    if seed is not None:    
        np.random.seed(seed + 10)
        dx = np.random.randint(range_low, range_high)
        np.random.seed(seed + 10 + 1)
        dy = np.random.randint(range_low, range_high)
        
        np.random.seed(seed + 20)
        x = np.random.randint(0, width - dx + 1)
        np.random.seed(seed + 20 + 1)
        y = np.random.randint(0, height - dy + 1)
       
    else:
        dx = np.random.randint(range_low, range_high)
        dy = np.random.randint(range_low, range_high)
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
    
    imgs_crop = []
    for img in imgs: 
        img_tmp = img[y:(y+dy), x:(x+dx)]
        imgs_crop.append(img_tmp)
    return imgs_crop

def crop_generator(batches, crop_length):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield (batch_crops, batch_y)

def crop_image_only_outside(label, img, tol=0, pad=256):
    # img is 2D image data
    # tol  is tolerance
    label = np.pad(label, ((pad, pad), (pad, pad)), 'constant', constant_values=0)
    img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=0)
    mask = label > tol
    m, n = label.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax() - pad, n-mask0[::-1].argmax() + pad
    row_start, row_end = mask1.argmax() - pad, m-mask1[::-1].argmax() + pad
    return (label[row_start:row_end,col_start:col_end], img[row_start:row_end,col_start:col_end])
        
def random_crop_batch(ipimglist, 
                      iplabellist, 
                      opfolder, 
                      label, 
                      crop_size, 
                      crop_per_image, 
                      crop_outside = False,
                      seed=None):
    '''
    Takes images in the input folder("ipfolder") and randomly crop the images in batch, and 
    save to the output folder("opfolder"). The range of cropping size can be defined by 
    "random_size_range". "crop_per_image" defines the amount of images generated from 
    each inputs. 
    '''
    
    # create the file list
    imglist = ipimglist[label]
    labellist = iplabellist[label]
    
    total_img_count = len(imglist) * crop_per_image

    
    id_count = 1
    
    # iterate through each files
    for idx in trange(len(imglist)): 
        # load the raw images
        img_tmp = imread(imglist[idx], as_gray=True)
        
        # load the labeled images
        label_tmp = imread(labellist[idx], as_gray=True)
        # plt.imshow(label_tmp)
        
        # Incase there are labels bigger than 1
        label_tmp_array = label_tmp > 0 

        
        if crop_outside: 
            label_tmp_array, img_tmp_array = crop_image_only_outside(label_tmp_array, img_tmp)
        else: 
            img_tmp_array = img_tmp
            label_tmp_array = label_tmp_array
        
        # print(label_tmp_array.shape)
        # plt.imshow(img_tmp_array)
        # plt.imshow(label_tmp_array)
        
        # while subimg_count < (crop_per_image + 1):        
        for i in range(crop_per_image):
            # crop the image by a give value
            imgs_crop = random_crop([img_tmp_array, label_tmp_array], crop_size, seed=seed)
            img_crop = imgs_crop[0]
            label_crop = imgs_crop[1]
            # plt.imshow(label_crop)
            
            # percentage = np.sum(label_crop)/(crop_size[0] * crop_size[1])
            
            # save image
            pil_img_crop = Image.fromarray(img_crop)
            pil_label_crop = Image.fromarray(label_crop)

            id_name = str(id_count)
            pil_img_crop.save(os.path.join(opfolder, 'images', label, id_name.zfill(4) + '.tif'))
            pil_label_crop.save(os.path.join(opfolder, 'labels', label, id_name.zfill(4) + '.tif'))

            id_count += 1
         
            if seed is not None:
                seed += 1
                        
def random_crop_batch_old(ipfolder, opfolder, label, random_size_range, crop_per_image, seed=None):
    '''
    Takes images in the input folder("ipfolder") and randomly crop the images in batch, and 
    save to the output folder("opfolder"). The range of cropping size can be defined by 
    "random_size_range". "crop_per_image" defines the amount of images generated from 
    each inputs. 
    '''
    
    # create the file list
    imglist = glob.glob(os.path.join(ipfolder, 'images', '*', '*.tif'), recursive=True)
    labellist = glob.glob(os.path.join(ipfolder, 'labels', '*', '*.tif'), recursive=True)
    
    '''
    print('First 5 filenames')
    print(imglist[:5])
    print('First 5 filenames')
    print(labellist[:5])
    '''

    id_count = 1

    # iterate through each files
    for idx, imgpath in enumerate(imglist): 
        # load the raw images
        img_tmp = imread(imglist[idx])

        # load the labeled images
        label_tmp = Image.open(labellist[idx])
        label_tmp_array = np.array(label_tmp)    
        
        for i in range(crop_per_image):
            # crop the image by a give value
            imgs_crop = random_crop([img_tmp, label_tmp_array], [256, 256], seed=seed)
        
            img_crop = Image.fromarray(imgs_crop[0])
            label_crop = Image.fromarray(imgs_crop[1])

            id_name = str(id_count)
            img_crop.save(os.path.join(opfolder, 'images', label, id_name.zfill(4) + '.tif'))
            label_crop.save(os.path.join(opfolder, 'labels', label, id_name.zfill(4) + '.tif'))

            if seed is not None:
                seed += 1
            
            id_count += 1
            
            
def create_crop_idx(img_size, target_size = (256, 256), overlap_fac = 0.1):
    '''
    img_size: the size(shape) of input image
    IMG_HEIGHT, IMG_WIDTH: height and width for the training network
    Cropping rule: from top-left to bottom-right, row first
    '''
    
    img_y = img_size[0]
    img_x = img_size[1]
    IMG_HEIGHT = target_size[0]
    IMG_WIDTH = target_size[1]
    overlap_fac = overlap_fac
    
    overlap_y = round(target_size[0] * overlap_fac) #overlap pixel
    step_y = target_size[0] - overlap_y # step size    
    
    y_step_c = math.ceil((img_y - IMG_HEIGHT) / step_y) + 1
    y_rem = (img_y - IMG_HEIGHT) % step_y 

    overlap_x = round(target_size[1] * overlap_fac) #overlap pixel
    step_x = target_size[1] - overlap_x # step size    
    
    x_step_c = math.ceil((img_x - IMG_WIDTH) / step_x) + 1
    x_rem = (img_x - IMG_WIDTH) % step_x 
    
    outputidx = np.empty((0, 6), int)
    
    ## create crop index
    # for imgidx_y in trange(y_step_c):  
    for imgidx_y in range(y_step_c): 
        if (imgidx_y+1)%y_step_c != 0:
            start_y = imgidx_y*step_y
            end_y = imgidx_y*step_y+IMG_HEIGHT
            for imgidx_x in range(x_step_c):    
                if (imgidx_x+1)%x_step_c != 0:
                    start_x = imgidx_x*step_x
                    end_x = imgidx_x*step_x+IMG_WIDTH
                    outputidx = np.append(outputidx, 
                                                np.array([[start_y, end_y, 
                                                           start_x, end_x, 
                                                           imgidx_y, imgidx_x]]), axis=0)
                else:
                    start_x = img_x-IMG_WIDTH
                    end_x = img_x
                    outputidx = np.append(outputidx, 
                                                np.array([[start_y, end_y, 
                                                           start_x, end_x, 
                                                           imgidx_y, imgidx_x]]), axis=0)
        else: 
            start_y = img_y-IMG_HEIGHT
            end_y = img_y
            for imgidx_x in range(x_step_c): 
                if (imgidx_x+1)%x_step_c != 0:
                    start_x = imgidx_x*step_x
                    end_x = imgidx_x*step_x+IMG_WIDTH
                    outputidx = np.append(outputidx, 
                                                np.array([[start_y, end_y, 
                                                           start_x, end_x, 
                                                           imgidx_y, imgidx_x]]), axis=0)
                else:
                    start_x = img_x-IMG_WIDTH
                    end_x = img_x
                    outputidx = np.append(outputidx, 
                                                np.array([[start_y, end_y, 
                                                           start_x, end_x, 
                                                           imgidx_y, imgidx_x]]), axis=0)  
    
    # print("Image Shape: {}, {}".format(img_y, img_x))
    # print("Patch size: {}, {}".format(IMG_HEIGHT, IMG_WIDTH))
    # print("Overlap Factor: {}".format(overlap_fac))
    # print("Step y: {}".format(step_y))
    # print("Step y count: {}".format(y_step_c))
    # print("Remainder in y: {}".format(y_rem))
    # print("Step x: {}".format(step_x))
    # print("Step x count: {}".format(x_step_c))
    # print("Remainder in x: {}".format(x_rem))
    
    return(outputidx)
            
            
def crop_to_patch(img, cropidx, target_size = (256, 256)):
    '''
    img: input image for cropping
    cropidx: index for cropping the image
    Var: [start_y, end_y, start_x, end_x]
    '''
    
    ## crop img
    outputimg = np.zeros((cropidx.shape[0], target_size[0], target_size[1]))
    # for idx in trange(outputimg.shape[0]):
    for idx in range(outputimg.shape[0]):
        start_y = cropidx[idx, 0]
        end_y = cropidx[idx, 1]
        start_x = cropidx[idx, 2]
        end_x = cropidx[idx, 3]
        outputimg[idx] = img[start_y:end_y, start_x:end_x]
   
    return(outputimg)


def construct_from_patch(img_stack: Any, 
                         cropidx: Any,
                         target_size: Union[List[int], Tuple[int]]):
    '''
    img: input image for cropping
    IMG_HEIGHT, IMG_WIDTH: height and width for the training network
    Cropping rule: from top-left to bottom-right, row first
    '''
    
    img_patch_y = img_stack.shape[1]
    img_patch_x = img_stack.shape[2]
    img_target_size_y = target_size[0]
    img_target_size_x = target_size[1]
    # reshape the gray image
    img_stack = np.reshape(img_stack, (img_stack.shape[0], img_stack.shape[1], img_stack.shape[2])) 
    
    # create the empty array
    img_stack_repos = np.full((img_stack.shape[0], img_target_size_y, img_target_size_x), np.nan)
    
    # for idx in trange(img_stack.shape[0]):
    for idx in range(img_stack.shape[0]):
        img_stack_repos[idx, cropidx[idx, 0]:cropidx[idx, 1], 
                        cropidx[idx, 2]:cropidx[idx, 3]] = img_stack[idx, :, :]
    
    outputimg = np.nanmean(img_stack_repos, axis = 0)
    
    # print("Patch Image Shape: {}, {}".format(img_patch_y, img_patch_x))
    # print("Target Image Size: {}, {}".format(img_target_size_y, img_target_size_x))
    return(outputimg)

def create_crop_idx_whole(img_size, target_size = (256, 256), overlap_fac = 0.1):
    '''
    img_size: the size(shape) of input image
    IMG_HEIGHT, IMG_WIDTH: height and width for the training network
    Cropping rule: from top-left to bottom-right, row first
    '''
    
    img_y = img_size[0]
    img_x = img_size[1]
    IMG_HEIGHT = target_size[0]
    IMG_WIDTH = target_size[1]
    overlap_fac = overlap_fac
    
    overlap_y = round(target_size[0] * overlap_fac) #overlap pixel
    step_y = target_size[0] - overlap_y # step size    
    
    y_step_c = math.ceil((img_y - IMG_HEIGHT) / step_y) + 1
    y_rem = (img_y - IMG_HEIGHT) % step_y 

    overlap_x = round(target_size[1] * overlap_fac) #overlap pixel
    step_x = target_size[1] - overlap_x # step size    
    
    x_step_c = math.ceil((img_x - IMG_WIDTH) / step_x) + 1
    x_rem = (img_x - IMG_WIDTH) % step_x 
    
    if (IMG_HEIGHT > img_y) | (IMG_WIDTH > img_x):
        print('At least one dimension of the image is smaller than patch size.')
        return
    
    outputidx = np.empty((0, 6), int)
    
    ## create crop index
    # for imgidx_y in trange(y_step_c):  
    for imgidx_y in range(y_step_c):
        if (imgidx_y+1)%y_step_c != 0:
            start_y = imgidx_y*step_y
            end_y = imgidx_y*step_y+IMG_HEIGHT
            for imgidx_x in range(x_step_c):
                if (imgidx_x+1)%x_step_c == 0:
                    start_x = img_x-IMG_WIDTH
                    end_x = img_x
                    outputidx = np.append(outputidx,
                                          np.array([[start_y, end_y,
                                                     start_x, end_x,
                                                     imgidx_y, imgidx_x]]), axis=0)
        else: 
            start_y = img_y-IMG_HEIGHT
            end_y = img_y
            for imgidx_x in range(x_step_c): 
                if (imgidx_x+1)%x_step_c != 0:
                    start_x = imgidx_x*step_x
                    end_x = imgidx_x*step_x+IMG_WIDTH
                    outputidx = np.append(outputidx, 
                                                np.array([[start_y, end_y, 
                                                           start_x, end_x, 
                                                           imgidx_y, imgidx_x]]), axis=0)
                else:
                    start_x = img_x-IMG_WIDTH
                    end_x = img_x
                    outputidx = np.append(outputidx, 
                                                np.array([[start_y, end_y, 
                                                           start_x, end_x, 
                                                           imgidx_y, imgidx_x]]), axis=0)
         
    # print("Image Shape: {}, {}".format(img_y, img_x))
    # print("Patch size: {}, {}".format(IMG_HEIGHT, IMG_WIDTH))
    # print("Overlap Factor: {}".format(overlap_fac))
    # print("Step y: {}".format(step_y))
    # print("Step y count: {}".format(y_step_c))
    # print("Remainder in y: {}".format(y_rem))
    # print("Step x: {}".format(step_x))
    # print("Step x count: {}".format(x_step_c))
    # print("Remainder in x: {}".format(x_rem))
    
    return(outputidx)


    