#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:24:57 2020

@author: dean
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import r2c_common as rc

sig2=0.25;
n_exps = 500
r2sims = np.linspace(0, 1, 5)
n = 4;m=362


ms = np.array([25, 50, 100, 150, 200])
ms = np.array([362,])
snrs = np.array([0.1, 0.25, 1])#pulling typical snr values

ncps = snrs*m*n#choose a desire ncp
mu2ys = ncps*sig2/n#adjust mu2y and consequently snr to match this ncp
snrs_new = ((mu2ys/m))/(sig2)#

mu2x = mu2ys[-1]
r2sims = np.linspace(0,1,5)
ql, qh = [0.05,0.95]

ncp_of_n2m_sims = m*n*((mu2ys/m)/sig2)

ncps = ncp_of_n2m_sims
mu2ys = (np.array(ncps)/(n))*sig2
_ = np.zeros([9, len(ncps), len(r2sims), len(ms), (n_exps)])

da = xr.DataArray(_, dims=['measure',  'snrs', 'r2s', 'm', 'exps',],
               coords=[['r2', 
                        'H&C_my_deriv',
                        'P&C',  
                        'H&C_their_deriv', 
                        'Y&D', 
                        'H&T', 
                        'S&L', 
                        'S&S',
                        'Zyl'], 
                       snrs, r2sims, ms, range(n_exps)])
da.attrs = {'n':n, 'm':m, 'mu2x':mu2x, 'sig2':sig2,  
            'n_exps':n_exps}
for k, mu2y in enumerate(mu2ys):
    r2cs = []
    r2s = []   
    for l, m in enumerate(ms):
        for j, r2 in enumerate(r2sims):
            theta = [r2, sig2, mu2y, mu2y, m, n]
            [x, y] = rc.pds_n2m_r2c(theta, n_exps, ddof=1)
            res = []
            for i in range(y.shape[0]):
                a_y = y[i]
                mod = np.zeros((len(x), 2))
                mod[:,0] = 1
                mod[:, 1] = x.squeeze()
                beta = np.linalg.lstsq(mod, a_y.mean(0), rcond=-1)[0]
                y_hat = np.dot(beta[np.newaxis], mod.T).squeeze()
                
                r2c, r2 = rc.r2c_n2m(x.T, a_y)  
                r2_pc = rc.r2_SE_corrected(x.squeeze(), a_y)
                r2_upsilon = rc.upsilon(y_hat, a_y)
                r2_hsu = rc.cc_norm_split(x.squeeze(), a_y)**2
                r2_yd = rc.r2_SB_normed(x.squeeze(), a_y)
                r2_sl = rc.normalized_spe(y_hat, a_y)
                r2_sc = rc.cc_norm(x.squeeze(), a_y)**2
                r2_zyl = rc.cc_norm_bs(x.squeeze(), a_y)**2
                res.append([np.double(r2.squeeze()), 
                            np.double(r2c.squeeze()), 
                            r2_pc, 
                            r2_upsilon, 
                            r2_yd, 
                            r2_hsu, 
                            r2_sl, 
                            r2_sc,
                            r2_zyl])
                
            res = np.array(res) 
            da[:,k,j, l, :] = res.T

            
#os.remove('./figs/fig_data/alt_n2m_meth_sim.nc')
da.to_netcdf('./alt_n2m_meth_sim.nc')
