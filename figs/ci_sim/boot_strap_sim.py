#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:42:07 2020

@author: dean
"""
import numpy as np
import r2c_common as rc
from tqdm import tqdm

def trunc_ud(u, l, x):
    x = np.array(x)
    x[x>u] = u
    x[x<l] = l
    return x

def trunc_d(d, x):
    x = np.array(x)
    x[x<d] = d
    return x

#boot strap simulation
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

r2 = 0.5
sig2 = 0.25
snr = 2
m = 30
n =  5
n_exps = 100
n_bs_samples = 500

ncps = snr*m
mu2y = mu2x = ncps*sig2
theta = [r2, sig2, mu2y, mu2y, m, n]
x, ys = pds_n2m_r2c(theta, n_exps, ddof=1)

res = []
for exp in tqdm(range(n_exps)):
    y = ys[exp]
    r2_er_obs = rc.r2c_n2m(x.squeeze(), y)[0]
    y_news = []
    for k in range(n_bs_samples):
        y_new = np.array([np.random.choice(y_obs, size=n) for y_obs in y.T]).T
        y_news.append(y_new)
    y_news = np.array(y_news)
    r2c_bs = rc.r2c_n2m(x.squeeze(), y_news)[0].squeeze()
    res.append([r2c_bs, r2_er_obs])
    
#%%

r2c_bs = np.array([a_res[0] for a_res in res])
cis = np.quantile(r2c_bs, [0.1, 0.9], 1).T
in_ci = (cis[:,0]<=r2)*(cis[:,1]>=r2)*(cis[:,0]!=cis[:,1])
#%%
cis = np.array([r[:2] for r  in res])
in_ci = (cis[:,0]<=r2s[i])*(cis[:,1]>=r2s[i])*(cis[:,0]!=cis[:,1])
in_cis.append(in_ci)
p = binom_test(np.sum(~in_ci), len(in_ci), p=alpha_targ)
plt.figure()
plt.title('Simulation of ' r'$r^2_{ER}$' 'CI \ntarget alpha=' + str(alpha_targ) + ', '+ 
      'sim. alpha=' + str(np.round(np.mean(~in_ci), 2))+
      ', p-val = ' + str(np.round(p, 2)) + '\n' +
      r'$\sigma^2 \sim U[0.1,2], \ d^2 \sim U[0.1, 2]$')

plt.plot(cis[:, 0], c='r')
plt.plot(cis[:, 1], c='g')
plt.scatter(range(len(in_ci)), ~np.array(in_ci), s=5, color='k')
#plt.scatter(range(len(in_ci)), np.array(cis[:,0]==cis[:,1]))
plt.ylim(0,1.1)
plt.yticks(r2s)
plt.grid()








    
