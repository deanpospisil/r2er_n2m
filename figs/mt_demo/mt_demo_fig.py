#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:48:40 2020

@author: dean
"""



import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
cwd = os.getcwd()
os.chdir('../../')
import r2c_common as r2c
os.chdir(cwd)
import pandas as pd


s = xr.open_dataarray('./mt_sqrt_spkcnt.nc')
snr = ((s.mean('trial_tot').var('dir')/
       s.var('trial_tot').mean('dir')))

theta = np.deg2rad(s.coords['dir'].values)[:,np.newaxis]
sin_mod = np.concatenate([np.cos(theta), np.sin(theta),np.ones((len(theta), 1))], -1)
coefs = np.linalg.lstsq(sin_mod, s.mean('trial_tot').values.T)[0]

pred = np.dot(sin_mod, coefs)
r2er = np.squeeze([r2c.r2c_n2m(a_pred, aunit.values)[0] 
              for a_pred, aunit in zip(pred.T, s)])

theta_fine = np.deg2rad(np.linspace(0,s.coords['dir'].values[-1]))[:,np.newaxis]
sin_mod_fine = np.concatenate([np.cos(theta_fine), 
                         np.sin(theta_fine),
                         np.ones((len(theta_fine), 1))], -1)

pred_fine = np.dot(sin_mod_fine,coefs)


os.chdir(cwd)
#os.chdir('../../')

p = pd.read_csv('./fits.csv', index_col=0)    
p['ciw'] = p['ul'] - p['ll']
pr = p.rank()
p['snr'] = snr
p=p.round(3)

pr = p.rank()
fig, ax = plt.subplots(nrows=2, ncols=2, 
                       sharex=True, sharey=True,
                       figsize=(4.5,4.5))

b_ind = pr.sum(1).argmax()
w_ind = (- pr['r2er']).argsort().values[-3]

uncertain_high = pr['ciw'].argsort().values[-2]
uncertain_low = pr['r2er'].argsort().values[-1]

ex_inds = [b_ind, w_ind,  uncertain_low, uncertain_high,]
k=0
for i in range(2):
    for j in range(2):
        ind = ex_inds[k]
        ax[i][j].plot(np.rad2deg(theta_fine), pred_fine[:,ind]**2/2)
        ax[i][j].errorbar(np.rad2deg(theta), 
                       (s**2/2).mean('trial_tot').values.T[:,ind],
                       yerr=(s**2/2).std('trial_tot').values.T[:,ind]/(10**0.5))
        ax[i][j].set_title('$\hat{r}^2_{ER}=$'+ str(np.round(p['r2er'][ind], 2))+
                        ', SNR=' + str(np.round(p['snr'][ind],2)) + 
                        '\nCI=[' + str(p['ll'][ind]) +', ' + str(p['ul'][ind])+']')
        
        if i==1 and j==0:
            ax[i][j].set_ylabel('spk/sec')
            ax[i][j].set_xlabel('Motion direction (deg)')
            ax[i][j].set_ylim(0,None)
            ax[i][j].legend(['Model', 'Neuron'])
        k+=1
fig.tight_layout()
fig.savefig('./mt_hilow_r2_er_ex.pdf')
#%% 
os.chdir(cwd)
#os.chdir('../../')

p = pd.read_csv('./fits.csv', index_col=0)    
p['ciw'] = p['ul'] - p['ll']
pr = p.rank()




pred_r2 = np.squeeze([r2c.r2c_n2m(a_pred, aunit.values)[1] 
              for a_pred, aunit in zip(pred.T, s)])

        
fig, ax = plt.subplots(nrows=1, ncols=1, 
                       sharex=False, sharey=True,
                       figsize=(8,2))
thresh=0.25
ms = 3
p_low_ciw = p[p['ciw']<thresh]
ax.errorbar(np.arange(len(p_low_ciw['r2er'])), p_low_ciw['r2er'], 
                   yerr=[-p_low_ciw['ll']+p_low_ciw['r2er'],
                                                p_low_ciw['ul']-p_low_ciw['r2er']],
            linestyle='none', marker='_', c='k', alpha=1, ms=ms)
ax.errorbar(np.arange(len(p_low_ciw['r2er'])), p_low_ciw['r2er'], 
                   yerr=[-p_low_ciw['ll']+p_low_ciw['r2er'],
                                                p_low_ciw['ul']-p_low_ciw['r2er']],
            linestyle='none', marker='_', c='r', alpha=1, ms=ms, lw=0)
p_hi_ciw = p[p['ciw']>thresh]

x = np.arange(len(p_low_ciw['r2er']), len(p_low_ciw['r2er'])+len(p_hi_ciw['r2er']))
ax.errorbar(x, p_hi_ciw['r2er'], 
                   yerr=[-p_hi_ciw['ll']+p_hi_ciw['r2er'],
                                                p_hi_ciw['ul']-p_hi_ciw['r2er']],
            linestyle='none', marker='_', c='k', alpha=1, ms=ms)
ax.errorbar(x, p_hi_ciw['r2er'], 
                   yerr=[-p_hi_ciw['ll']+p_hi_ciw['r2er'],
                                                p_hi_ciw['ul']-p_hi_ciw['r2er']],
            linestyle='none', marker='_', c='r', alpha=1, ms=ms, lw=0)

ax.set_yticks([0, 0.25, 0.5, 0.75, 1]);
ax.set_ylim(-0.1,1.2)
ax.set_xticks([0, len(p_low_ciw['r2er']), len(p['r2er'])])
ax.set_xticklabels(['CI length <' + str(thresh), 'CI length >'+ str(thresh), ''], 
                   ha='left' )

ax.set_title('MT neuron fit to sinusoid model')
ax.set_ylabel('$\hat{r}^2_{ER}$');
ax.grid()
fig.savefig('./mt_r2er_hilo_ci.pdf')
#%%
fig, ax = plt.subplots(nrows=1, ncols=1, 
                       sharex=False, sharey=True,
                       figsize=(6,3))
inds = [b_ind, w_ind]

ax.plot(p_low_ciw['r2'], p_low_ciw['r2er'], 
            linestyle='none', marker='.', fillstyle='none', c='k', alpha=1)
ax.plot([0,1], [0,1], linestyle='--', alpha=0.5)

ax.plot(p_hi_ciw['r2'], p_hi_ciw['r2er'], 
            linestyle='none', marker='.', fillstyle='none', c='r', alpha=1)
ax.plot([0,1], [0,1], linestyle='--', alpha=0.5)

ax.axis('square')

ax.set_xlim(0,1.1);ax.set_ylim(0,1.1);
ax.grid();
ax.set_xticks([0, 0.25, 0.5, 0.75, 1]);
ax.set_yticks([0, 0.25, 0.5, 0.75, 1]);
ax.set_xlabel('$\hat{r}^2$');ax.set_ylabel('$\hat{r}^2_{ER}$');

fig.tight_layout()
fig.savefig('./mt_r2er_vs_r2.pdf')


#%%
ax.errorbar(p_low_ciw['r2'], p_low_ciw['r2er'], 
            yerr=[-p_low_ciw['ll']+p_low_ciw['r2er'],
                                                p_low_ciw['ul']-p_low_ciw['r2er']],
            linestyle='none', marker='.', fillstyle='none', c='k', alpha=0.4)

ax.errorbar(p_low_ciw['r2'][inds], p_low_ciw['r2er'][inds], 
            yerr=[-p_low_ciw['ll'][inds]+p_low_ciw['r2er'][inds],
                                                p_low_ciw['ul'][inds]-p_low_ciw['r2er'][inds]],
            linestyle='none', marker='.',  c='r', alpha=1)