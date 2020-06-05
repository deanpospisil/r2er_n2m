#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:07:32 2020

@author: dean
"""


import xarray as xr 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


ds = xr.open_dataset('ci_sim_data.nc')
params = ds['params'].attrs
cis = ds['cis']
r2ers = cis.coords['r2er']

cis_vals = cis.values
cis_vals[cis_vals>1] = 1
cis_vals[cis_vals<0] = 0

cis[...] = cis_vals

cis = cis[...,:,:]

in_ci = ((cis.sel(ci='ll')<=r2ers)*(cis.sel(ci='ul')>=r2ers)*(cis.sel(ci='ll')!=1)*(cis.sel(ci='ll')!=1)*(cis.sel(ci='ul')!=0))
ms = in_ci.mean('exp')

n_exps = len(in_ci.coords['exp'])
ses = np.sqrt(ms*(1-ms)/n_exps)


s = in_ci.sum('exp')
ps = np.zeros(s.shape)
for i in range(s.shape[0]):
    for j in range(s.shape[1]):
        p = stats.binom_test(s[i,j], n_exps, p=0.8)
        ps[i,j] = p
        

to_hi = ((cis.sel(ci='ll')>r2ers)+(cis.sel(ci='ll')==1)).sum('exp')
to_lo = ((cis.sel(ci='ul')<r2ers)+(cis.sel(ci='ul')==0)).sum('exp')
ps_uneven = np.zeros(s.shape)
for i in range(s.shape[0]):
    for j in range(s.shape[1]):
        p = stats.binom_test([to_hi[i,j], to_lo[i,j]], n_exps, p=0.5)
        ps_uneven[i,j] = p
        
rej = ps<(0.01/(np.product(s.shape)))
rej_uneven = ps_uneven<(0.01/(np.product(s.shape)))

colors = ['c', 'r', 'g']
for i in range(3): 
    plt.errorbar(r2ers, ms[:,i], yerr=ses[:,i], color=colors[i])
plt.legend(['Non-parametric bootstrap', 'Parametric bootstrap', 'Hybrid bayes'])

for i in range(3):
    plt.scatter(r2ers[rej[:,i]], -0.05 + (ms)[:,i][rej[:,i]], marker='*', edgecolors=colors[i], facecolors='none')
    plt.scatter(r2ers[rej_uneven[:,i]], 0.05+(ms)[:,i][rej_uneven[:,i]], edgecolors=colors[i], facecolors=colors[i])

plt.ylabel('Fraction CI contain true ${r}^2_{ER}$')
plt.xlabel(r'${r}^2_{ER}$')
plt.grid()
#%%
plt.figure(figsize=(10,3))
sub_samp = 1
r2er_ind = 3
r2er = r2ers[r2er_ind]
yticks = np.linspace(0,1,5)
plt.subplot(131)
r2_er = r2ers[r2er_ind]
cis[0, r2er_ind, ::sub_samp, 0].plot()
cis[0, r2er_ind, ::sub_samp, 1].plot()
plt.ylim(0,1.1)
plt.plot([0,500], [r2er,r2er])
plt.ylabel(r'${r}^2_{ER}$')
plt.xlabel('simulation')
plt.legend(['lower CI', 'upper CI', 'True $r^2_{ER}$'])
plt.title('Non parametric\nbootstrap')
plt.subplot(132)

cis[1, r2er_ind, ::sub_samp, 0].plot()
cis[1, r2er_ind, ::sub_samp, 1].plot()
plt.title('')
plt.ylim(0,1.1)
plt.plot([0,500], [r2er,r2er])
plt.gca().set_yticklabels([])
plt.gca().set_xticklabels([])
plt.ylabel('')
plt.xlabel('')
plt.title('Parametric\nbootstrap')



plt.subplot(133)

cis[2, r2er_ind, ::sub_samp, 0].plot()
cis[2, r2er_ind, ::sub_samp, 1].plot()
plt.ylim(0,1.1)
plt.plot([0,500], [r2er,r2er])
plt.gca().set_yticklabels([])
plt.gca().set_xticklabels([])
plt.ylabel('')
plt.xlabel('')
plt.title('Hybrid bayes')

#%%
cis[2, r2er_ind, ::sub_samp, 0].plot()
cis[2, r2er_ind, ::sub_samp, 1].plot()
plt.plot([0,500], [r2er,r2er])

#%%

to_hi = ((cis.sel(ci='ll')>r2ers)+(cis.sel(ci='ll')==1)).sum('exp')
to_lo = ((cis.sel(ci='ul')<r2ers)+(cis.sel(ci='ul')==0)).sum('exp')
ps_uneven = np.zeros(s.shape)
for i in range(s.shape[0]):
    for j in range(s.shape[1]):
        p = stats.binom_test([to_hi[i,j], to_lo[i,j]], n_exps, p=0.5)
        ps_uneven[i,j] = p

#%%
