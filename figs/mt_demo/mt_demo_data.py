#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:08:16 2020

@author: dean
"""


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
cwd = os.getcwd()
os.chdir('../../')
import r2c_common as r2c


import xarray as xr
from scipy.stats import spearmanr
import pandas as pd

#%%

load_dir = './data/v1dotd/'
fns = os.listdir(load_dir)
fns = [fn for fn in fns if '.nc' in fn ]
mt = [xr.open_dataarray(load_dir+fn).load() for  fn in fns]
fns = [fn.split('.')[0] for fn in fns]

os.chdir(cwd)

mt = xr.concat(mt, 'rec')
mt.coords['rec'] = range(len(fns))
mt.coords['nms'] = ('rec', [fn[:-1] for fn in fns])


mt_s = mt.sel(t=slice(0,2), unit=[0,1])
s = mt_s.sum('t', min_count=1)
s = s**0.5    
ind = ((~s.isnull()).sum('trial_tot')>9).prod('dir').prod('unit').values
s = s[(ind==1)] 
s = s.dropna('rec', how='all').dropna('trial_tot')
snr = ((s.mean('trial_tot').var('dir')/
       s.var('trial_tot').mean('dir')).prod('unit')**0.5)
nms = s.coords['nms']

ss = xr.concat([s[:, 0], s[:, 1]], 'rec')
ss.coords['rec'] = range(len(ss.coords['rec']))
ss.to_netcdf('./mt_sqrt_spkcnt.nc')

#%%
os.chdir(cwd)
s = xr.open_dataarray('./mt_sqrt_spkcnt.nc')
snr = ((s.mean('trial_tot').var('dir')/
       s.var('trial_tot').mean('dir')))

theta = np.deg2rad(s.coords['dir'].values)[:,np.newaxis]
sin_mod = np.concatenate([np.cos(theta), np.sin(theta),np.ones((len(theta), 1))], -1)
coefs = np.linalg.lstsq(sin_mod, s.mean('trial_tot').values.T, rcond=None)[0]

pred = np.dot(sin_mod, coefs)
r2er = np.squeeze([r2c.r2c_n2m(a_pred, aunit.values)[0] 
              for a_pred, aunit in zip(pred.T, s)])





dat = np.zeros((len(s.coords['rec']), 4))

p = pd.DataFrame(dat, columns=['r2er', 'r2', 'll', 'ul'])

for i in range(dat.shape[0]):
    x = pred[:, i]
    y = s[i].values
    ll, ul, r2c_hat_obs, alpha_obs = r2c.r2c_n2m_ci(x, y, 
                                                    alpha_targ=0.10, 
                                                    nr2cs=100)
    
    r2 = np.corrcoef(y.mean(0), x)[0,1]**2
    
    p.iloc[i] = [r2c_hat_obs, r2, ll, ul] 
#%%
os.chdir(cwd)
p.to_csv('./fits.csv')    

    
pd.read_csv('./fits.csv', index_col=0)    
    