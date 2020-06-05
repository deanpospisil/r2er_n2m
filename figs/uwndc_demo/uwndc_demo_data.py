#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:59:47 2020

@author: dean
"""
import numpy as np
import tensorflow as tf
import xarray as xr
import matplotlib.pyplot as plt
import os
cwd = os.getcwd()
os.chdir('../../')
import r2c_common as rc

model_dir = 'data/oleg_uwndc_sub/model/'
stim_path = 'data/oleg_uwndc_sub/stim.npy'# path to the file with images
image_size = 48# the model works with images 48x48
imgs = np.load(stim_path)# load images from the training dataset
# crop images to the size that the model accepts
crop = ((80 - image_size) // 2) if image_size < 80 else 0
imgs = imgs[:, crop:-crop, crop:-crop] if crop else imgs[50:]
# load the model
predict_fn = tf.contrib.predictor.from_saved_model(model_dir)
preds  = predict_fn({'image': imgs[:]})['spikes']

#load the data
r = xr.open_dataarray('data/oleg_uwndc_sub/full_spike_data.nc')
n_units = len(r.coords['unit'])
n_trial = len(r.coords['trial'])
rm = r.sel(t=slice(0.05,.35))
ns =  [rm.isel(unit=i).dropna('stim', how='all'
                       ).dropna('trial', how='any'
                       ).sum('t') 
                       for i in range(n_units)]
sigma2 = np.array([(ns[i]**0.5).var('trial').mean('stim').values 
                  for i in  range(n_units)])

#take mean variance neurons w/out  trials>1
one_trial_inds = sigma2==0
sigma2[one_trial_inds] = np.mean(sigma2[~one_trial_inds])

m_test = 50 # held out data
n_sim = 50
size = (n_units, 2, n_trial-1, n_sim)
res = xr.DataArray(np.zeros(size), 
                   dims=['unit', 'est', 'n', 'sim'],
                   coords=[range(size[0]), 
                           ['r2c', 'r2'], 
                           range(2, n_trial+1), 
                           range(n_sim)])
res[...] = np.nan
#%%
'''
from tqdm import tqdm
for sim in tqdm(range(n_sim)):#run this many simulations
    for n in range(2, n_trial+1):#how many repeats to simulate
        for i in range(n_units):
            neur_dat = (ns[i]**0.5)[:m_test].values
            unit_tot_trial = neur_dat.shape[-1]
            pred = preds[:m_test, i]
            
            if unit_tot_trial>=n:#if there are enough trials
                #choose with replacement random trials from each stimulis
                neur_dat_samp = np.array([neur_dat[k, 
                                          np.random.choice(unit_tot_trial, n)] 
                                          for k in range(m_test)])
                r2c, r2 = rc.r2c_n2m(pred, 
                                        neur_dat_samp.T)
                res.loc[i, :, n, sim] = np.array([r2c, r2]).squeeze()
                


os.chdir(cwd)
os.remove('./sim_vary_n.nc')
res.to_netcdf('./sim_vary_n.nc')

'''
#%%
size = (n_units, 4)

res_ci = xr.DataArray(np.zeros(size), dims=['unit', 'm', ],
              coords=[range(size[0]), ['r2c', 'r2', 'll', 'ul']])

res_ci[...] = np.nan

for i in range(n_units):
    neur_dat = (ns[i]**0.5)[:m_test].copy(deep=True).values
    n_trial = neur_dat.shape[-1]
    
    pred = preds[:m_test, i]
    
    if n_trial>1:
        ll, ul, r2c_hat_obs, alpha_obs = rc.r2c_n2m_ci(pred, neur_dat.T, alpha_targ=0.10, nr2cs=100)
        r2 = np.corrcoef(pred, neur_dat.mean(-1))[0,1]**2
        res_ci.loc[i,:] = np.array([r2c_hat_obs, r2, ll, ul]).squeeze()

#%%
os.chdir(cwd)     
res_ci.to_netcdf('./uwndc_oleg_ci_mtest' + str(m_test) + '.nc')
