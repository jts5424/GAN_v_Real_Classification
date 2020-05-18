# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle
from inputs import *
from import_data import *
from plotting import *
from recon import *
from learn import *


#%%
# Module One: Getting data ready for ML Input

# use import data libray to import images and convert to analysis capable structures
# from 'import_data'
gan, real = run_import(gan_folder,real_folder,img_height,img_width,max_items)

# assign data to dictionary to save data
data_arrays = to_dict([gan,real],['gan','real'])

# save data
with open('data_arrays', 'wb') as f:
    pickle.dump(data_arrays, f)

#%%
# Module 2: Singular Value decomposition (Preprocessing, not necessary for ML)
    
# open data from previous module  
file = open('data_arrays', 'rb')
data = pickle.load(file)  

# compute SVD to understand proper orthogonal mode variance / dimensionality reduction oportunity
all_data = np.append(data['gan'],data['real'], 0)
u_gan, s_gan, vh_gan = np.linalg.svd(data['gan'], full_matrices=False)
u_real, s_real, vh_real = np.linalg.svd(data['real'], full_matrices=False)
u_all, s_all, vh_all = np.linalg.svd(all_data, full_matrices=False)

# Save SVD Data to dictionary
gan_svd_dict = to_dict([u_gan,s_gan,vh_gan],['U','S','V'])
real_svd_dict = to_dict([u_real,s_real,vh_real],['U','S','V'])
all_svd_dict = to_dict([u_all,s_all,vh_all],['U','S','V'])

Data_SVD = {'Gan':gan_svd_dict, 'Real':real_svd_dict, 'All Data':all_svd_dict}

with open('Data_SVD', 'wb') as f:
    pickle.dump(Data_SVD, f)




#%%
# Module 3: Plot Gan vs Real singular values to investigate for differences in optimal dimensionality reduction
dims = 200
x = np.arange(0,dims)
gan_cumsum_var = np.power(s_gan[0:x.max()+1],2).cumsum()/np.power(s_gan[0:x.max()+1],2).cumsum().max()
real_cumsum_var = np.power(s_real[0:x.max()+1],2).cumsum()/np.power(s_real[0:x.max()+1],2).cumsum().max()
# from 'plotting.py'
dual_plot(x,gan_cumsum_var,real_cumsum_var,'GAN','Real','# of Singular Values','% Cumulitive Energy','Gan vs Real Singular Value Spectrum','Sing_Values.png')   
# This particular dataset does not show a clear difference between Gan vs Real singular values 


#%%
# Module 4: Use Plot from Module 3 to choose reconstruction ranks and calculate difference relative to original images
# If difference in singular values is apparent, reconstruction relative to original image between gan and real can lead to better classification
ranks = np.asarray([10,20,30,40,50,60,70,80,90,100,150,200,300,400])
# from 'recon.py'
trunc_error_gan, trunc_error_real = recon(all_data,u_all,s_all,vh_all,ranks)
# from 'plotting.py'
dual_plot(ranks,trunc_error_gan,trunc_error_real,'GAN','Real','Reconstruction Rank','Normed Error Relative to Full Rank','Gan vs Real Reconstruction Fidelity','recon.png')    
# Does not yield greatly improved classification because singular values are similar


#%%
file = open('data_arrays', 'rb')
data = pickle.load(file) 

gan = data['gan']
real = data['real']

gan_ml = np.append(gan,np.ones([len(gan),1],float),1)
real_ml = np.append(real,np.zeros([len(real),1],float),1)
all_data_ml = np.append(gan_ml,real_ml, 0)

# define classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver = 'newton-cg')
# from 'learn.py'
auc, thresh = classify(classifier,all_data_ml)
auc_avg = np.mean(auc)
thresh_avg = np.mean(thresh)
