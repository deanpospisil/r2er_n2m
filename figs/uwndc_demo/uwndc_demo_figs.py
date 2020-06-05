#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:21:45 2020

@author: dean
"""


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
cwd = os.getcwd()



res = xr.open_dataarray('./sim_vary_n.nc')
plt.figure(figsize=(3,3))
 
ind = [10, 9, 14, ]
#ind = range(n_units)
c = ['b', 'r', 'g']*20
the_ns = res.coords['n'].values
for unit_ind, ac in zip(ind, c):
    unit_result = res[unit_ind].dropna('n')
    plt.errorbar(unit_result.coords['n'], unit_result[0].mean('sim', skipna=True), 
                 yerr=unit_result[0].std('sim', skipna=True), 
                 c=ac)
    
    plt.errorbar(unit_result.coords['n'], unit_result[1].mean('sim', skipna=True), 
                 #yerr=res[a_ind,1,:].std('sim', skipna=True), 
                 c=ac, fmt='--')
plt.yticks(np.linspace(0,1,5))
plt.xticks(np.arange(2,7,1))
plt.xlim(1.5, 7)

plt.ylim(-0.1,1)
plt.grid()

plt.legend([r'$\hat{r}^2$', r'$\hat{r}^2_{ER}$'][::-1], loc='top right', 
           ncol=2)
plt.xlabel('Trials (n)')
plt.ylabel(r'$r^2$')
plt.title('Mean ' + r'$\hat{r}^2$' +' and '+ r'$\hat{r}^2_{ER}$'  )
plt.tight_layout()

plt.savefig('./uwndc_r2_vs_r2er_n.pdf')


#%%
import pandas as pd
res_ci = xr.open_dataarray('./uwndc_oleg_ci.nc')
csv = pd.read_csv('./train.csv')

is_single_unit = np.array([int(nm.split('_')[-1])==1 for nm in csv.columns[1:]])
yerr = np.array([-res_ci[:, 2] + res_ci[:, 0], res_ci[:, 3] - res_ci[:, 0] ])

plt.figure(figsize=(3,3))
for i in [0,1]:
    plt.errorbar(res_ci.values[is_single_unit==i,1], res_ci.values[is_single_unit==i, 0], 
                 yerr=yerr[:, is_single_unit==i], fmt='o')
plt.legend(['multi-unit', 'single-unit'], fontsize=8)

plt.axis('square');plt.grid();
plt.xlim(-0.1,1.1);plt.ylim(-0.1,1.1);
plt.plot([0,1],[0,1], c='k');
xticks = np.linspace(0, 1, 5)
plt.xticks(xticks)
plt.yticks(xticks)
plt.ylabel('$\hat{r}^2_{ER}$')
plt.xlabel('$\hat{r}^2$')
plt.title('DCNN prediction\nperformance of V4')
plt.tight_layout()
plt.savefig('./uwndc_r2_vs_r2er.pdf')

#%%
from scipy.stats import ttest_ind
plt.figure(figsize=(4,2))
res_ci_sort = res_ci[np.argsort(is_single_unit)]
is_single_unit_sort = np.sort(is_single_unit)
ind =  ~res_ci_sort[:,0].isnull().values
res_ci_sort = res_ci_sort[ind]
is_single_unit_sort = is_single_unit_sort[ind]

yerr_sort = np.array([-res_ci_sort[:, 2] + res_ci_sort[:, 0], 
                      res_ci_sort[:, 3] - res_ci_sort[:, 0] ])

fitss = []
c = ['blue', 'orange']
for i in range(2):
    fits = res_ci_sort[is_single_unit_sort==i,0].values
    err_bars = yerr_sort[:,is_single_unit_sort==i]
    pop_mean = np.mean(fits)
    pop_sd = np.std(fits)
    
    plt.plot()
    if i==0:
        x = np.arange(len(is_single_unit_sort[is_single_unit_sort==0]))
    else:
        x = np.arange(len(is_single_unit_sort[is_single_unit_sort==0]),
                      len(is_single_unit_sort[is_single_unit_sort==0])+
                      len(is_single_unit_sort[is_single_unit_sort==1]))
    
    plt.errorbar(x, fits,  yerr=err_bars, fmt='o', c=c[i])
    plt.plot([x.min(), x.max()], [pop_mean, pop_mean], c=c[i])
    fitss.append(fits)

    
#plt.legend(['multi-unit', 'single-unit'], fontsize=12)

plt.ylim(0,1)
plt.yticks(xticks)
plt.xticks([0, 7.5, 13])
plt.gca().set_xticklabels([])
plt.annotate('Multi-unit', (2, 0.05))
plt.annotate('Single-unit', (9, 0.05))
plt.grid()

print(ttest_ind(fitss[0], fitss[1], equal_var=False))
plt.ylabel(r'$\hat{r}^2_{{ER}}$')

plt.title('Individual and population\nDNN to V4 fit estimates')
plt.tight_layout()


plt.savefig('./uwndc_r2er_pop_&_indiv.pdf')


