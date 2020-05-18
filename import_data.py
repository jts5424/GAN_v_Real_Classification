# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 17:46:27 2020

@author: JTSDellLaptop
"""

import os
import glob

from PIL import Image
import numpy as np

# Sets folder location variables
def inputs(folder):
    
    img_dir = folder
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)
    
    return files


# Converts images in folder location specified into data array
# Reshapes images into rows (each image is a row in the final array)
# Converts RGB into grayscale [RGB is 3 dimensional], need grayscale [1 Dim] for ML implementation)
# Resize data for efficiency (this loses data so comparison with small sample of original [larger data] should be investigated)
def get_data(files,height,width):
    
    data = []
    
    for f1 in files:
        image_array = np.array(Image.open(f1).convert('L').resize((height,width))).reshape(height*width)
        data.append(image_array)

    data = np.asarray(data,float)
    
    return data


# Import images from directory using functions from above
def run_import(gan_folder,real_folder,dim1,dim2,max_items):

    # acquire path variables
    gan_files = inputs(gan_folder)
    real_files = inputs(real_folder)
    
    # import image data and convert to numerical structures for analysis
    gan = get_data(gan_files,dim1,dim2) # gan images
    # Choose a random subset of gan images equal to the number of real images (results in same number of gan and real images)
    gan = gan[np.random.choice(gan.shape[0], max_items, replace=False), :] # choose a random number of samples equal to the number of real images available
    real = get_data(real_files,dim1,dim1) # real Images
    
    return gan, real
