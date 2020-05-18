# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 23:19:51 2020

@author: JTSDellLaptop
"""
import numpy as np

def recon(data,u,s,vh,dims):
    
    trunc_error_gan = []
    trunc_error_real = []
    dim_1 = data.shape[0]/2
    dim_2 = data.shape[1]

    
    for i in range(len(dims)):
    
        recon = np.dot(u[:,:dims[i]], np.dot(np.diag(s[:dims[i]]), vh[:dims[i], :]))
        trunc_error_gan.append(np.linalg.norm(recon[:int(dim_1),:] - data[:int(dim_1),:]))
        trunc_error_real.append(np.linalg.norm(recon[int(dim_1):,:] - data[int(dim_1):,:]))

    trunc_error_gan = np.asarray(trunc_error_gan,float)/(np.linalg.norm(data[:int(dim_1),:]))
    trunc_error_real = np.asarray(trunc_error_real,float)/(np.linalg.norm(data[int(dim_1):,:]))
    
    return trunc_error_gan, trunc_error_real