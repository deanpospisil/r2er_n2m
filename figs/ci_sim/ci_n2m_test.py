#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 09:44:59 2020

@author: dean
"""

import numpy as np
import r2c_common as rc
import xarray as xr
from scipy.stats import ttest_1samp
from tqdm import tqdm
import matplotlib.pyplot as plt



#neuron to model ci test
n_exps = 50
sig2=0.25;
m=362;n=4;


#max_spike = np.array([0.25, 2 ])
#mu2ys = m*max_spike/4
snrs = np.array([0.1,])#pulling typical snr values
#snrs = np.array([0.01, 0.05, 1])#pulling typical snr values

#snrs = 0.25
ncps = snrs*362*4#choose a desire ncp
mu2ys = ncps*sig2/n#adjust mu2y and consequently snr to match this ncp
snrs_new = ((mu2ys/m))/(sig2)#

mu2x = mu2y = mu2ys
r2sims = np.linspace(0, 1, 1)

ncp_of_n2m_sims = m*n*((mu2ys/m)/sig2)


_ = np.zeros([2, len(mu2ys), len(r2sims), n_exps])
da = xr.DataArray(_, dims=['measure', 'snr', 'r2s', 'sim'],
               coords=[['ll', 'ul'],
                       snrs, r2sims, range(n_exps)])
da.attrs = {'n':n, 'm':m, 'mu2x':mu2x, 'sig2':sig2,  
            'n_exps':n_exps} 


alpha_targ=0.10
alpha_obss = []

for r2 in r2sims:
    print(r2)
    for snr, mu2y in zip(snrs, mu2ys):
        theta = [r2, sig2, mu2y, mu2y, m, n]
        x,y = rc.pds_n2m_r2c(theta, n_exps, ddof=1);
        x = x.squeeze()[np.newaxis,np.newaxis] 
        for i in range(n_exps):
            ll, ul, r2c_hat_obs, alpha_obs = rc.r2c_n2m_ci(x[0], y[i], 
                                                           alpha_targ=alpha_targ, 
                                                           nr2cs=50);
            da.loc[:, snr, r2, i] = [ll, ul]
            alpha_obss.append(alpha_obs)

#%%
hits = da[0,...,:].copy(deep=True)
for r2 in r2sims:
    ll = da.sel(r2s=r2)[0]
    ul = da.sel(r2s=r2)[1]
    hit = (ll<=r2)*(ul>=r2)
    hits.loc[...,r2,:] = hit
    print('')
    print(r2)
    print('')
    print(hit)
    print('')
    print(ll)
    print('')
    print(ul)
    
#%%
    
i = 0
print(hits[0, i].mean())
print(ttest_1samp(hits[0, i], 0.9))

#%%
#
r2=1
da.sel(r2s=r2)[0,0].plot.line(x='sim')
da.sel(r2s=r2)[1,0].plot.line(x='sim')
plt.yticks(np.linspace(0,1,5))
#da.sel(r2s=r2)[1].plot.line(x='sim')
#da.to_netcdf('first_ci_sim_may18th.nc')

    
