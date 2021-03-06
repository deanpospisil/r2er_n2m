#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:15:01 2020

@author: dean
"""
from scipy.stats import ncx2
from scipy.stats import multivariate_normal
from scipy.stats import binom_test

import r2c_common as rc

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.stats import norm
def trunc_ud(u, l, x):
    x = np.array(x)
    x[x>u] = u
    x[x<l] = l
    return x

def trunc_d(d, x):
    x = np.array(x)
    x[x<d] = d
    return x

def trunc_ud_scalar(u, l, x):
    if x>u:
        return u
    elif x<l:
        return l
    return x

def s2_hat_obs_n2m(y):
    s2_hat = (np.mean(np.var(y, -2, ddof=1, keepdims=True), -1, keepdims=True))
    return s2_hat


def mu2_hat_obs_n2m(y):
    
    ym = np.mean(y, -2, keepdims=True)
    y_ms = ym - np.mean(ym, -1, keepdims=True)
    
    mu2y = np.sum(y_ms**2, -1, keepdims=True)
    
    return mu2y

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

def r2c_n2m_ci_from_post(trace, n, m, r2c_hat_obs, alpha_targ=0.10, nr2cs=50, 
                     nr2c_hat=1000):
    
    sig2s = trace[0]
    mu2ys = trace[1]*m
    sample_inds = np.random.choice(len(sig2s), size=nr2c_hat, replace=False)
    n_exps = 1
    ress = []
    r2s =  np.linspace(0, 1, nr2cs)
    for r2 in r2s:
      res = []
      for i in range(nr2c_hat):
        i = sample_inds[i]
        theta = [r2, sig2s[i], mu2ys[i], mu2ys[i], m, n]
        x, y = rc.pds_n2m_r2c(theta, n_exps, ddof=1)
        res.append(rc.r2c_n2m(x.squeeze(), y)[0])
      ress.append(np.array(res).squeeze())
    ress = np.array(ress)

    tol = alpha_targ
    alpha_obs = np.mean(ress>r2c_hat_obs, 1)

    ll_dif = np.abs(alpha_obs-alpha_targ/2)
    ll_ind = np.argmin(ll_dif)
    if ll_dif[ll_ind]<tol:
        ll = r2s[ll_ind]
    else:
        ll = 0
    
    ul_dif = np.abs(alpha_obs-(1-alpha_targ/2))
    ul_ind = np.argmin(ul_dif)
    if ul_dif[ul_ind]<tol:
        ul = r2s[ul_ind]
    else:
        ul=1

    return ll, ul, alpha_obs


def sample_post_s2_d2(x, n_samps, trunc_sig2=[0, np.inf], trunc_d2=[0, np.inf]):
    n, m = x.shape
    hat_sig2 = x.var(0, ddof=1).mean(0)#central chisquared
    hat_d2 = x.mean(0).var(0, ddof=1)#non-central chisquared
    #make sure they don't go outside range
    hat_sig2 = trunc_ud_scalar(trunc_sig2[1], trunc_sig2[0], hat_sig2)
    hat_d2 = trunc_ud_scalar(trunc_d2[1], trunc_d2[0], hat_d2)

    sig2_curr = hat_sig2
    d2_curr = hat_d2


    df = m-1
    scale = sig2_curr/(n*(m-1))
    nc = (d2_curr*m)/(sig2_curr/n)
    vd2 = ncx2.var(df, nc, loc=0, scale=scale)
    
    df = m*(n-1)
    nc = 0
    scale = sig2_curr/(m*(n-1))
    vs2 = ncx2.var(df, nc, scale=scale)
    
    
    accept = np.zeros(n_samps)
    trace = np.zeros((2, n_samps))
    u = np.random.uniform(0, 1, n_samps)
    for i in (range(n_samps)):
        sig2_cand, d2_cand = np.random.normal(loc=[sig2_curr, d2_curr], 
                                              scale=[vs2**0.5, vd2**0.5])
        
        scale = sig2_cand/(n*(m-1))
        fd2_cand = (ncx2._pdf(hat_d2*(scale**-1), 
                            df = m-1, 
                            nc = (d2_cand*m)/(sig2_cand/n))*(scale**-1)*
                    int((d2_cand<=trunc_d2[1])*
                        (d2_cand>=trunc_d2[0])))
        
        scale = sig2_curr/(n*(m-1))
        fd2_curr = ncx2._pdf(hat_d2*(scale**-1), 
                            df=m-1, 
                            nc=(d2_curr*m)/(sig2_curr/n))*(scale**-1)
        
        scale = sig2_cand/(m*(n-1))
        fs2_cand = (ncx2._pdf(hat_sig2*(scale**-1), 
                             df=m*(n-1), 
                             nc=0)*(scale**-1)*
                    int((sig2_cand<=trunc_sig2[1])*
                        (sig2_cand>=trunc_sig2[0])))
        scale = sig2_curr/(m*(n-1))
        fs2_curr = ncx2._pdf(hat_sig2*(scale**-1), 
                             df=m*(n-1), 
                             nc=0)*(scale**-1)
        
        a = (fs2_cand*fd2_cand)/(fs2_curr*fd2_curr)
        if a>=1:
            sig2_curr = sig2_cand
            d2_curr = d2_cand
            accept[i] = 1
        else:
            if u[i] < a:
                accept[i] = 1
                sig2_curr = sig2_cand
                d2_curr = d2_cand
                    
        
        trace[:, i] = [d2_curr, sig2_curr]
    return trace, np.mean(accept)


def get_emp_dist_r2er(r2c_check, r2c_hat_obs, trace, m, n,
                    p_thresh=0.01, n_r2c_sims = 100):
    sig2_post = trace[1]#trial-to-trial variance
    d2m_post = trace[0]*m#dynamic range

    sample_inds = np.random.choice(len(sig2_post), 
                               size=n_r2c_sims, 
                               replace=True)#randomly sample from post-dist
    
    res = np.zeros(n_r2c_sims)
    for j in range(n_r2c_sims):
      k = sample_inds[j]
      theta = [r2c_check, sig2_post[k], d2m_post[k], d2m_post[k], m, n]
      x, y = pds_n2m_r2c(theta, 1, ddof=1)
      res[j] = (rc.r2c_n2m(x.squeeze(), y)[0]).squeeze()
    return res
def find_sgn_p_cand(r2c_check, r2c_hat_obs, alpha_targ, trace, m, n,
                    p_thresh=0.01, n_r2c_sims = 100):
    
    #checks on the cdf of r2c whether the observed r2c_hat_obs is
    #above the desired cdf value +1
    #null the desired cdf value
    #below
    z_thresh = norm.ppf(1.-p_thresh)
    res = get_emp_dist_r2er(r2c_check, r2c_hat_obs,  trace, m, n,
                    p_thresh=p_thresh, n_r2c_sims = n_r2c_sims)

      
    count = (res<r2c_hat_obs).sum()

    z = ((count - alpha_targ*n_r2c_sims)/
         (n_r2c_sims*alpha_targ*(1-alpha_targ))**0.5)
    
    sgn_p_cand = np.nan
    if z>z_thresh:
        sgn_p_cand = 1
    elif -z>z_thresh:
        sgn_p_cand = -1
    else:
        sgn_p_cand = 0
    return sgn_p_cand, res


def find_cdf_pos(r2c_hat_obs, alpha_targ, trace, m, n, n_splits=6,
                    p_thresh=1e-6, n_r2c_sims = 100, 
                    int_l=0, int_h=1):
    
    sgn_p_cand_h, res = find_sgn_p_cand(r2c_check=int_h, 
                                 r2c_hat_obs=r2c_hat_obs, 
                                 alpha_targ=alpha_targ, 
                                 trace=trace, m=m, n=n,
                                 p_thresh=p_thresh, n_r2c_sims=n_r2c_sims)
    
    sgn_p_cand_l, res = find_sgn_p_cand(r2c_check=int_l, 
                                 r2c_hat_obs=r2c_hat_obs, 
                                 alpha_targ=alpha_targ, 
                                 trace=trace, m=m, n=n,
                                 p_thresh=p_thresh, n_r2c_sims=n_r2c_sims)
    
    if sgn_p_cand_h==1 or sgn_p_cand_h==0:
        return int_h, res
    if sgn_p_cand_l==-1 or sgn_p_cand_l==0:
        return int_l, res
    
    #int_hs = [int_h,]
    #int_ls = [int_l,]
    #sgn_p_cands = [0,]
    for split in range(n_splits):
        
        #c_cand = (int_h-int_l)/2. + int_l
        c_cand = np.random.uniform(int_l, int_h)#just to keep it off grid
        
        sgn_p_cand, res = find_sgn_p_cand(r2c_check=c_cand, 
                                 r2c_hat_obs=r2c_hat_obs, 
                                 alpha_targ=alpha_targ, 
                                 trace=trace, m=m, n=n,
                                 p_thresh=p_thresh, n_r2c_sims=n_r2c_sims)
        if sgn_p_cand==-1:
            int_h = c_cand
        elif sgn_p_cand==1:
            int_l = c_cand
        if sgn_p_cand==0:
            return c_cand, res
        
    return c_cand, res
def solve_for_x_lin_interp(x0, x1, y0, y1, y):
    
    x = y*(x1-x0)/(y1-y0) - y0 + x0
    
    if x>x1:
        x=x1
    elif x<x0:
        x=x0
    
    return x


def get_ci(x, y, n_r2c_sims=1000, alpha_targ=0.1, p_thresh=0.01, n_splits=6,
           trunc_sig2=[0, np.inf], trunc_d2=[0, np.inf]):
    #get confidence intervals
    n, m = y.shape
    r2c_hat_obs = rc.r2c_n2m(x.squeeze(), y)[0]
    trace, p = sample_post_s2_d2(y, n_samps=2000, 
                                 trunc_sig2=trunc_sig2, trunc_d2=trunc_d2)# get posterior dist of params
    
    ul, ul_alpha = find_cdf_pos(r2c_hat_obs, alpha_targ/2, 
                              trace, m, n, n_splits=n_splits,
                              p_thresh=p_thresh, n_r2c_sims=n_r2c_sims, 
                              int_l=0, int_h=1)
    ll, ll_alpha = find_cdf_pos(r2c_hat_obs, 1-alpha_targ/2, 
                              trace, m, n, n_splits=n_splits,
                              p_thresh=p_thresh, n_r2c_sims = n_r2c_sims, 
                              int_l=0, int_h=1)
    
    return ll, ul, r2c_hat_obs, trace, ll_alpha, ul_alpha

#%%
x0 = 1
x1 = 2
y0 = 1
y1 = 2

y = 1.9

x = solve_for_x_lin_interp(x0, x1, y0, y1, y)


plt.plot([x0, x1], [y0, y1])
plt.scatter(x, y)

#%%
sig2 = 0.25
snr = 1
m = 40
n =  4
ncps = snr*m
mu2y = mu2x = ncps*sig2

n_splits = 10
n_r2c_sims = 1000
p_thresh = 0.001
n_r2c_sims = 300
alpha_targ = 0.2
trunc_sig2=[0.1, 2]
trunc_d2=[0.1, 2]
n_exps = 200
r2 = 0.5

#%%
theta = [r2, sig2, 1, mu2y, m, n] 
x, y = pds_n2m_r2c(theta, 1, ddof=1)
#get confidence intervals
n, m = y.shape
r2c_hat_obs = rc.r2c_n2m(x.squeeze(), y)[0].squeeze()
trace, p = sample_post_s2_d2(y, n_samps=2000, 
                             trunc_sig2=trunc_sig2, trunc_d2=trunc_d2)# get posterior dist of params
#%%
r2c_check = r2c_hat_obs
res = get_emp_dist_r2er(r2c_check, r2c_hat_obs, trace, m, n,
                    p_thresh=0.01, n_r2c_sims = 1000)

r2_qs = np.quantile(res, [0, 0.33, 0.5, 0.66, 1])

eval_frac = []
for r2c_check in r2_qs:
    res = get_emp_dist_r2er(r2c_check, r2c_hat_obs, trace, m, n,
                    p_thresh=0.01, n_r2c_sims = 1000)
    eval_frac.append((res>r2c_hat_obs).mean())

#%%
ll, ul, r2c_hat_obs, trace, ll_alpha, ul_alpha = get_ci(x, y,  
                                                    n_r2c_sims=n_r2c_sims, 
                                                    alpha_targ=alpha_targ, 
                                                    p_thresh=p_thresh,
                                                    n_splits=n_splits,
                                                    trunc_sig2=trunc_sig2, 
                                                    trunc_d2=trunc_d2)


#%%

p_thresh=1e-2
n_r2c_sims = 5000
int_l=0
int_h=1
alpha_targ = 0.9
n_splits = 30
sgn_p_cand_h, res_h = find_sgn_p_cand(r2c_check=int_h, 
                             r2c_hat_obs=r2c_hat_obs, 
                             alpha_targ=alpha_targ, 
                             trace=trace, m=m, n=n,
                             p_thresh=p_thresh, n_r2c_sims=n_r2c_sims)

sgn_p_cand_l, res_l = find_sgn_p_cand(r2c_check=int_l, 
                             r2c_hat_obs=r2c_hat_obs, 
                             alpha_targ=alpha_targ, 
                             trace=trace, m=m, n=n,
                             p_thresh=p_thresh, n_r2c_sims=n_r2c_sims)



if sgn_p_cand_h==1 or sgn_p_cand_h==0:
    print(int_h, (res_h<r2c_hat_obs).sum()/n_r2c_sims, 'a')
if sgn_p_cand_l==-1 or sgn_p_cand_l==0:
    print( int_l, (res_l<r2c_hat_obs).sum()/n_r2c_sims, 'b')



#int_hs = [int_h,]
#int_ls = [int_l,]
#sgn_p_cands = [0,]
for split in range(n_splits):
    plt.figure()
    c_cand = (int_h-int_l)/2. + int_l
    c_cand = np.random.uniform(int_l, int_h)
    
    # c_cand = solve_for_x_lin_interp(int_l, int_h, 
    #                                 (res_l>r2c_hat_obs).sum()/n_r2c_sims, 
    #                                 (res_h>r2c_hat_obs).sum()/n_r2c_sims, 
    #                                 alpha_targ)
    
    plt.hist(res_h, cumulative=True, density=True, histtype='step', color='g', bins=1000)
    plt.hist(res_l, cumulative=True, density=True, histtype='step', color='r', bins=1000)
    plt.plot([int_l, int_l], [0,1], c='r')
    plt.plot([int_h, int_h], [0,1], c='g')
    plt.plot([c_cand, c_cand], [0,1], c='b')
    plt.xlim(0,1)

    #plt.plot([int_l, int_h],
             #[(res_l>r2c_hat_obs).sum()/n_r2c_sims, (res_h>r2c_hat_obs).sum()/n_r2c_sims])
    #plt.scatter(c_cand, alpha_targ)
    
    plt.plot([r2c_hat_obs,r2c_hat_obs], [0,1], c='k')
    sgn_p_cand, res = find_sgn_p_cand(r2c_check=c_cand, 
                             r2c_hat_obs=r2c_hat_obs, 
                             alpha_targ=alpha_targ, 
                             trace=trace, m=m, n=n,
                             p_thresh=p_thresh, n_r2c_sims=n_r2c_sims)
    if sgn_p_cand==-1:
        int_h = c_cand
        res_h = res
    if sgn_p_cand==1:
        int_l = c_cand
        res_l = res
    if sgn_p_cand==0:
        #print(c_cand, res, 'c')
        break
    
print((res>r2c_hat_obs).sum()/ n_r2c_sims)
print(split/n_splits)

#%%
r2s = np.linspace(0,1,4)
ress = []
for r2 in tqdm(r2s):
    ys = []
    sig2 = np.random.uniform(trunc_sig2[0], trunc_sig2[1], size=n_exps);
    d2 = np.random.uniform(trunc_d2[0], trunc_d2[1], size=n_exps);
    for a_d2, a_sig2 in zip(sig2, d2):
        theta = [r2, a_sig2, 1, a_d2*m, m, n] 
        x, y = pds_n2m_r2c(theta, 1, ddof=1)
        ys.append(y)
    ys = np.array(ys)
    res = []
    for i in tqdm(range(n_exps)):
        ll, ul, r2c_hat_obs, trace, ll_alpha, ul_alpha = get_ci(x, ys[i],  
                                                            n_r2c_sims=n_r2c_sims, 
                                                            alpha_targ=alpha_targ, 
                                                            p_thresh=p_thresh,
                                                            n_splits=n_splits,
                                                            trunc_sig2=trunc_sig2, 
                                                            trunc_d2=trunc_d2)
        
        res.append([ll, ul, r2c_hat_obs, trace, ll_alpha, ul_alpha])
    ress.append(res)
    
#%%
in_cis = []
for i, res in enumerate(ress):
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

    plt.legend(['upper ci', 'lower ci'][::-1])

