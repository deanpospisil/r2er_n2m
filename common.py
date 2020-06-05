#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:33:47 2020

@author: dean
"""

import numpy as np

def r2c_n2m(x, y):
    """approximately unbiased estimator of R^2 between the expected values. 
        of the rows of x and y. Assumes x is fixed and y has equal variance across 
        trials and observations.
    Parameters
    ----------
    x : numpy.ndarray
        m observations model predictions
    y : numpy.ndarray
        m observations by n trials array of data
  
    Returns
    -------
    r2c : an estimate of the r2 between the expected values 
    --------
    """
    n, m = np.shape(y)[-2:]
    sigma2 = np.mean(np.var(y, -2, ddof=1, keepdims=True), -1, keepdims=True)
    x_ms = x - np.mean(x, keepdims=True)
    
    y = np.mean(y, -2, keepdims=True)
    
    y_ms = y - np.mean(y, (-1), keepdims=True)
    
    xy2 = np.sum((x_ms*y_ms), -1, keepdims=True)**2
    
    x2 = np.sum(x_ms**2, -1, keepdims=True)
    y2 = np.sum(y_ms**2, -1, keepdims=True)
    
    x2y2 = x2*y2
    

    ub_xy2 = xy2 - sigma2/n * x2
    ub_x2y2 = x2y2 - (m-1)*sigma2/n*x2
    
    r2c = ub_xy2/ub_x2y2
    
    return r2c, xy2/x2y2

   
def trunc_ud(u, l, x):
    x = np.array(x)
    x[x>u] = u
    x[x<l] = l
    return x

def trunc_d(d, x):
    x = np.array(x)
    x[x<d] = d
    return x

def pds_n2m_r2c(theta, n_exps, ddof=1):
    [r2, sig2, mu2x, mu2y, m, n] = theta
    r2 = trunc_ud(1, 0, r2)
    mu2y = trunc_d(0, mu2y)
    angle = np.arccos(r2**0.5)
   
    if not angle.shape==():
        angle = angle[np.newaxis]
        mu2y = mu2y[np.newaxis]        
    s = np.linspace(0, 2*np.pi, int(m))[:, np.newaxis]
   
    xu = np.cos(s + angle*0)
    yu = np.cos(angle + s)

    yu = (mu2y**0.5)*(yu/(yu**2.).sum(-2)**0.5)

    y = np.random.normal(loc=yu, scale=sig2**0.5,
                         size=(n_exps,) + (int(n),) + yu.shape )
   
    x = xu
    y = np.rollaxis(y,-1,0).squeeze()
   
    return x,y