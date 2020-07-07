#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:15:54 2020

@author: dean
"""


import os
os.chdir('../../')

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
#%%
plt.figure(figsize=(4,3))
#plt.subplot(211)
my = 4
mx = np.pi*2
leg_fs = 9
ms = 6
r2 = 1
m = 10
n = 4
sig2 = 0.5
angle = np.arccos(r2**0.5)
s = np.linspace(0, 2*np.pi- 2*np.pi/m, m)
mu2x= 4
mu2y = 6


xu = np.sin(s)
yu = np.sin(angle + s)
xu = (mu2x**0.5)*(xu/(xu**2.).sum(-1)**0.5)
yu = (mu2y**0.5)*(yu/(yu**2.).sum(-1)**0.5)
xu = xu-2*xu.min()
yu = yu - 2*yu.min()

x = np.random.normal(loc=xu, scale=sig2**0.5,
                     size=(int(n),) + xu.shape )
y = np.random.normal(loc=yu, scale=sig2**0.5,
                     size= (int(n),) + yu.shape )
#plt.title('Neuron tuning curve, estimated tuning curve, and model')

plt.plot(s, xu, 'o-r', lw=2,)
plt.plot(s, yu, 'o-g', lw=2, alpha=0.5)
plt.errorbar(s, yu, yerr=(sig2**0.5)/(n**0.5), alpha=0.5, c='g')
#plt.plot(s, yu)
plt.plot(s, y.mean(0), 'o--g', mfc='None')
s = np.broadcast_to(s, np.shape(y))
#plt.plot(s.ravel(), x.ravel(), '.', c='r', alpha=0.5)
#plt.plot(s.ravel(), y.ravel(), 'o', c='g', mfc='none', alpha=0.7, lw=2)
#plt.ylabel(r'$ \sqrt{{ \mathrm{spike \ count}}}$')
#plt.xlabel('stimuli phase (rad.)')
plt.xticks([0, np.pi, np.pi*2])
plt.gca().set_xticklabels([r'',r'',r'',])

legend_elements = [
                   Line2D([0], [0], marker='o', color='r', label='Model prediction',
                          markerfacecolor='r', markersize=ms, lw=2),
                   Line2D([0], [0], marker='o', color='g', lw=2,
                          label='Neuron tuning curve',
                          mfc='g', markersize=ms),
                    Line2D([0], [0], lw=2, marker='o', color='g', label='Estimated tuning curve',
                          mfc='None', markersize=ms, linestyle='--'),]

plt.xlim(-2*np.pi/m, mx);plt.ylim(0,my)

plt.legend(handles=legend_elements, fontsize=leg_fs, loc='lower left' )

plt.ylabel(r'$ \sqrt{{ \mathrm{spike \ count}}}$')
plt.xlabel('Stimuli phase (radians)')
plt.xticks([0, np.pi, np.pi*2])
plt.gca().set_xticklabels([r'$0$',r'$\pi$',r'$2\pi$',])
plt.yticks([0, 1, 2, 3, 4])
r2 = np.round(np.corrcoef(y.mean(0), yu)[0,1]**2, 2)
plt.text(np.pi, 3, 'True ' r'$r^2_{ER}=1$' '\nEstimated ' r'$\hat{r}^2=$' 
         + str(r2))
plt.tight_layout()
plt.savefig('./figs/n2m_ex_fig.pdf')  
#%% initial demo of method
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6,3))
quantiles = [0.1, 0.9]
res = xr.open_dataarray('./figs/r2er_n2m_sim/vary_r2er.nc')

ax[0].hist(res.sel(r2=1, est='r2').squeeze())
ax[0].hist(res.sel(r2=1, est='r2c').squeeze())
ax[0].plot([1,1], [0,680], c='k')
ax[0].set_ylabel('Count');

ax[0].legend(['True $r^2$','Naive $\hat{r}^2$','Corrected $\hat{r}^2_{ER}$', ],
             fontsize=8)

ax[0].set_xlim(0,1.2);ax[0].set_ylim(0,700);
ax[0].set_xlabel('Estimate');


resq = res.quantile(quantiles, 'sim')
resm = res.mean('sim')
m = resm.squeeze()
q = resq.squeeze()

for est  in ['r2', 'r2c']:
    ax[1].errorbar(x=m.coords['r2'], y=m.sel(est=est),
                 yerr=[q[0].sel(est=est)-m.sel(est=est),
                       m.sel(est=est)-q[1].sel(est=est)] )
ticks = np.linspace(0,1,5)
ax[1].set_xlim(0,1.1);ax[1].set_ylim(0,1.1);
ax[1].set_xticks(ticks);ax[1].set_yticks(ticks);
ax[1].axis('square');ax[1].grid()
ax[1].set_ylabel('Estimate');ax[1].set_xlabel('True $r^2_{ER}$');

ax[1].plot([0,1], [0,1], c='k', lw=3);

fig.tight_layout()
fig.savefig('./figs/r2er_n2m_sim/demo_r2_er.pdf')

#%% r2er r2 as function of SNR, n, m
r2 = 0.75
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(6,3))
i = 0

yticks = [0, 0.25, 0.5, 0.75, 1]
res = xr.open_dataarray('./figs/r2er_n2m_sim/vary_SNR.nc')
resq = res.quantile(quantiles, 'sim')
resm = res.mean('sim')
m = resm.squeeze()
q = resq.squeeze()
#for i, r2 in enumerate(r2s):
for est  in ['r2', 'r2c']:
    ax[i].errorbar(x=m.coords['snr'].values, y=m.sel(est=est, r2=r2),
                 yerr=[q[0].sel(est=est, r2=r2)-m.sel(est=est, r2=r2),
                       m.sel(est=est, r2=r2)-q[1].sel(est=est, r2=r2)] )
ax[i].semilogx()
ax[i].set_xticks([0.1, 1, 10]);ax[i].set_yticks(yticks)
ax[i].set_ylim(0,1);
ax[i].set_xlabel('SNR')
ax[i].grid()
ax[i].axhline(0.75, c='k')
ax[i].legend(['True $r^2$','Naive $\hat{r}^2$','Corrected $\hat{r}^2_{ER}$'],
             fontsize=8)
ax[i].set_title('n=' + str(int(m.coords['ns'].values))+
                ', m=' + str(int(m.coords['ms'].values)))

ax[i].set_ylabel('$r^2$')
i+=1
res = xr.open_dataarray('./figs/r2er_n2m_sim/vary_n.nc')
resq = res.quantile(quantiles, 'sim')
resm = res.mean('sim')
m = resm.squeeze()
q = resq.squeeze()
#for i, r2 in enumerate(r2s):
for est  in ['r2', 'r2c']:
    ax[i].errorbar(x=m.coords['ns'].values, y=m.sel(est=est, r2=r2),
                 yerr=[q[0].sel(est=est, r2=r2)-m.sel(est=est, r2=r2),
                       m.sel(est=est, r2=r2)-q[1].sel(est=est, r2=r2)] )
ax[i].semilogx()
ax[i].set_xticks([1, 10,]);ax[i].set_yticks(yticks)
ax[i].set_yticklabels([])
ax[i].set_ylim(0,1)
ax[i].set_xlabel('n')
ax[i].grid()
ax[i].axhline(0.75, c='k')
ax[i].set_title('SNR=' + str(np.double(m.coords['snr'].values))+
                ', m=' + str(int(m.coords['ms'].values)))

i+=1       
res = xr.open_dataarray('./figs/r2er_n2m_sim/vary_m.nc')
resq = res.quantile(quantiles, 'sim')
resm = res.mean('sim')
m = resm.squeeze()
q = resq.squeeze()
#for i, r2 in enumerate(r2s):
for est  in ['r2', 'r2c']:
    ax[i].errorbar(x=m.coords['ms'].values, y=m.sel(est=est, r2=r2),
                 yerr=[q[0].sel(est=est, r2=r2)-m.sel(est=est, r2=r2),
                       m.sel(est=est, r2=r2)-q[1].sel(est=est, r2=r2)] )
ax[i].semilogx()
ax[i].set_xticks([10, 1000]);ax[i].set_yticks(yticks)
ax[i].set_yticklabels([])
ax[i].set_ylim(0,1)
ax[i].set_xlabel('m')
ax[i].grid()
ax[i].axhline(0.75, c='k')
ax[i].set_title('n=' + str(int(m.coords['ns'].values)) + 
                ', SNR=' + str(np.double(m.coords['snr'].values))
                )


fig.tight_layout()
fig.savefig('./figs/r2er_n2m_sim/r2_r2er_vs_SNR_n_m.pdf')
