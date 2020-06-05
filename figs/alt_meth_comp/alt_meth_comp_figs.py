#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:35:33 2020

@author: dean
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

da = xr.open_dataarray('./alt_n2m_meth_sim.nc')


da = da.sel(measure=['r2', 
                        'H&C_my_deriv',
                        'P&C',  
                        'H&C_their_deriv', 
                        'Y&D', 
                        'H&T', 
                        'S&L', 
                        'S&S',
                        'Zyl'])

nms = [r'$\hat{r}^2$', 
        r'$ \hat{r}^2_{ER}$',
        r'$ \hat{r}^2  (1-\frac{SE^2}{SS_{total}})$',  
        r'$ \Upsilon $', 
        r'$\hat{r}^2_{norm-split-SB}$', 
        r'$CC^2_{norm-split}$', 
        r'$SPE_{norm}$', 
        r'$CC^2_{norm-SP}$',
        r'$CC^2_{norm-PB}$']

da.coords['measure'] = nms


xticks_labels = ['0', '0.25', '0.5', '0.75', '1','1.25', '1.5', '1.75', '2']
xticks = [0,.25,.5,.75,1,1.25,1.5,1.75,2]
plt.figure(figsize=(8,3.5))
colors = ['r', 'm']
for j, snr_ind in enumerate([1,2]):
    plt.subplot(1,2,j+1)

    
    for i, r2er_ind in enumerate(([1,3])):
        snr = da[:,  snr_ind, -r2er_ind, -1].coords['snrs'].values
        m_stim = da[:,  snr_ind, -r2er_ind, -1].coords['m'].values
        snr_val = np.round(snr, 2)

        m = da[:,  snr_ind, -r2er_ind, -1].mean('exps')
        sd = da[:, snr_ind, -r2er_ind, -1].std('exps')
        
        mse = ((da[:,1, -1, -1]-1)**2).mean('exps')
        ind = np.argsort(mse.values)[::-1]
        plt.errorbar(y=range(len(m)), x=m[ind], xerr=sd[ind], c=colors[i])
        plt.xlim(0,2)
        plt.gca().set_yticks(range(len(nms)))
        plt.gca().set_xticks(xticks)

        if j==0:
            #plt.title('Simulation based comparison\nof neuron-to-model fit estimators')
            plt.gca().set_yticklabels(np.array(nms)[ind], rotation=0)
            plt.gca().set_xticklabels(xticks, rotation=0)

            plt.xlabel('Estimated $r^2_{ER}$')
            plt.ylabel('Estimators (ordered by MSE)')
            plt.legend(['$r^2_{ER}=1$', '$r^2_{ER}=0.5$'], 
                       title='')
            plt.text(1.4,0.5, 'SNR=' + str(snr_val) + '\nn=4, m='+str(m_stim))


        else:
            plt.gca().set_yticklabels('', rotation=0)
            plt.gca().set_xticklabels(xticks, rotation=0)

            plt.text(1.4,0.5, 'SNR=' + str(snr_val) + '\nn=4, m='+str(m_stim))
    plt.grid()


plt.tight_layout()
plt.savefig('./alt_n2m_simulation.pdf')

#%% evaluation on real data




   