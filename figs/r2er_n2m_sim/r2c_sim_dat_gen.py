#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:03:09 2020

@author: dean
"""

import os
os.chdir('../../')

import r2c_common as rc
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def SNR_to_mu2(snr, sig2, m, n):
    ncp = snr*m*n#choose a desire ncp
    mu2 = ncp*sig2/n
    return mu2

def get_r2c_sim(coords):
    dims = ['est', 'sim', 'r2', 'sig2', 'snr', 'ms', 'ns', ]
    res = xr.DataArray(np.zeros(tuple(len(c) for c in coords)),
             coords=coords,
             dims=dims)
    res = res.stack(c=dims[2:])
    for i in range(res.shape[-1]):
        [r2, sig2, snr, m, n ]= res[:, :, i].coords['c'].values.tolist()
        mu2y = SNR_to_mu2(snr, sig2, m, n)
        theta = [r2, sig2, mu2y, mu2y, m, n]
        x,y = rc.pds_n2m_r2c(theta, n_exps, ddof=1)
        x = x.squeeze()[np.newaxis,np.newaxis]
        r2c, r2 = rc.r2c_n2m(x, y)   
        
        res[..., i] = np.array([r2, r2c]).squeeze()
    res = res.unstack()
    
    return res
            
n_exps = 2000
measures = ['r2', 'r2c', ]

#first we go with changing r2s
r2s = np.linspace(0, 1, 5)
ms = [362,]
ns = [4,]
snrs = [0.5,]
sig2s = [0.25,]
coords = [measures, range(n_exps), r2s, sig2s, snrs, ms, ns]
res = get_r2c_sim(coords)
res.to_netcdf('./figs/r2er_n2m_sim/vary_r2er.nc')

#%%
#changing SNR
r2s = np.linspace(0, 1, 5)
ms = [362,]
ns = [4,]
snrs = np.logspace(-1,1,10)
sig2s = [0.25,]
coords = [measures, range(n_exps), r2s, sig2s, snrs, ms, ns]
res = get_r2c_sim(coords)
res.to_netcdf('./figs/r2er_n2m_sim/vary_snr.nc')
#%%
#changing n
r2s = np.linspace(0, 1, 5)
ms = [362,]
ns = [2, 4, 8, 16, 32]
snrs = [0.5, ]
sig2s = [0.25, ]
coords = [measures, range(n_exps), r2s, sig2s, snrs, ms, ns]
res = get_r2c_sim(coords)
res.to_netcdf('./figs/r2er_n2m_sim/vary_n.nc')

#%%
#changing m
r2s = np.linspace(0, 1, 5)
ms = [20, 40, 80, 160, 320, 640, 1280]
ns = [4,]
snrs = [0.5, ]
sig2s = [0.25, ]
coords = [measures, range(n_exps), r2s, sig2s, snrs, ms, ns]
res = get_r2c_sim(coords)
res.to_netcdf('./figs/r2er_n2m_sim/vary_m.nc')

