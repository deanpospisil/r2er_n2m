#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:31:18 2020

@author: dean
"""

import sys
sys.path.append('../../')
from scipy.stats import ncx2
import common as rc
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from scipy import stats
import xarray as xr 
import multiprocessing as mp


def percentile(x,q):
    inds = np.argsort(x)
    ind = int((len(x)-1)*q/100.)
    val = x[inds[ind]]
    return val
    

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

def mu2_hat_unbiased_obs_n2m(y):
    n,m = y.shape[-2:]
    sig2_hat = s2_hat_obs_n2m(y)
    ym = np.mean(y, -2, keepdims=True)
    y_ms = ym - np.mean(ym, -1, keepdims=True)
    mu2y = np.sum(y_ms**2, -1, keepdims=True) - (m-1)*sig2_hat/n
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
    #get the observed statistics
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
        
        sig2_cand = trunc_ud_scalar(trunc_sig2[1], trunc_sig2[0], sig2_cand)
        d2_cand = trunc_ud_scalar(trunc_d2[1], trunc_d2[0], d2_cand)
        
        scale = sig2_cand/(n*(m-1))
        fd2_cand = (ncx2._pdf(hat_d2*(scale**-1), 
                            df = m-1, 
                            nc = (d2_cand*m)/(sig2_cand/n))*(scale**-1))
        
        scale = sig2_curr/(n*(m-1))
        fd2_curr = ncx2._pdf(hat_d2*(scale**-1), 
                            df=m-1, 
                            nc=(d2_curr*m)/(sig2_curr/n))*(scale**-1)
        
        
        scale = sig2_cand/(m*(n-1))
        fs2_cand = ncx2._pdf(hat_sig2*(scale**-1), 
                             df=m*(n-1), 
                             nc=0)*(scale**-1)
                    
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
    #null the desired cdf value 0
    #below the desired cdf value -1
    z_thresh = norm.ppf(1.-p_thresh)#the z-value
    res = get_emp_dist_r2er(r2c_check, r2c_hat_obs,  trace, m, n,
                    p_thresh=p_thresh, n_r2c_sims = n_r2c_sims)

      
    count = (res<r2c_hat_obs).sum()

    z = ((count - alpha_targ*n_r2c_sims)/
         (n_r2c_sims*alpha_targ*(1-alpha_targ))**0.5)#get z score for this count
    
    sgn_p_cand = np.nan
    if z>z_thresh:
        sgn_p_cand = 1
    elif -z>z_thresh:
        sgn_p_cand = -1
    else:
        sgn_p_cand = 0
    return sgn_p_cand, res


def find_cdf_pos(r2c_hat_obs, alpha_targ, trace, m, n, n_splits=6,
                 p_thresh=1e-2, n_r2c_sims = 100, int_l=0, int_h=1):
    
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
    
    #we first check if 1 has a proportion less than r2er_obs
    #which is greater or essentially equal to the target then you cannot increase
    # the upper limit than 1 and so you stop.
    if sgn_p_cand_h==1 or sgn_p_cand_h==0:
        return int_h, res
    
        
    #we first check if 0 has a proportion less than r2er_obs
    #which is less than or essentially equal to the target then you cannot increase
    # the upper limit than 1 and so you stop.
    if sgn_p_cand_l==-1 or sgn_p_cand_l==0:
        return int_l, res

    for split in range(n_splits):
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

def get_hyb_bayes_ci(x, y, n_r2c_sims=1000, alpha_targ=0.1, p_thresh=0.01, n_splits=6,
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

def get_npbs_ci(x, y, alpha_targ, n_bs_samples=1000):
    y_bs = []
    for k in range(n_bs_samples):
        _ = np.array([np.random.choice(y_obs, size=n) for y_obs in y.T]).T
        y_bs.append(_)
    y_bs = np.array(y_bs)
    r2c_bs = rc.r2c_n2m(x.squeeze(), y_bs)[0].squeeze()
    
    ci = np.quantile(r2c_bs, [alpha_targ/2, 1 - alpha_targ/2]).T
    return ci

def get_pbs_ci(x, y, alpha_targ, n_pbs_samples=1000):
    r2c_hat_obs = rc.r2c_n2m(x.squeeze(), y)[0].squeeze()
    hat_sig2 = s2_hat_obs_n2m(y)
    hat_mu2y = mu2_hat_unbiased_obs_n2m(y)
    theta = [r2c_hat_obs, hat_sig2, 1, hat_mu2y, m, n] 
    x_new, y_new = pds_n2m_r2c(theta, n_pbs_samples, ddof=1)
    
    r2c_pbs = rc.r2c_n2m(x_new.squeeze(), y_new)[0].squeeze()
    ci = np.quantile(r2c_pbs, [alpha_targ/2, 1 - alpha_targ/2]).T

    return ci

def get_pbs_bca_ci(x, y, alpha, n_bs_samples):
    #need percentiles alpha and 1-alpha
    z_alpha = stats.norm.ppf(alpha/2.)
    z_1_alpha = stats.norm.ppf(1-alpha/2.)
    m = y.shape[-1]
    n = y.shape[-2]
    r2c_hat_obs = rc.r2c_n2m(x.squeeze(), y)[0].squeeze()
    hat_sig2 = s2_hat_obs_n2m(y)
    hat_mu2y = mu2_hat_unbiased_obs_n2m(y)
    theta = [r2c_hat_obs, hat_sig2, 1, hat_mu2y, m, n] 
    x_new, y_new = pds_n2m_r2c(theta, n_bs_samples, ddof=1)
    
    r2c_pbs = rc.r2c_n2m(x_new.squeeze(), y_new)[0].squeeze()
    
    #jack knife
    jack_r2c = []
    for i in range(m):
        jack_y = np.array([y[:,j] for j in range(m) if i!=j]).T
        jack_x = np.array([x[j] for j in range(m) if i!=j])
        _ = rc.r2c_n2m(jack_x, jack_y)[0].squeeze()
        jack_r2c.append(_.squeeze())
        
    jack_r2c = np.array(jack_r2c)
    jack_r2c_dot = jack_r2c.mean()
    
    #need bias correction factor
    bias_factor = np.mean(r2c_pbs<r2c_hat_obs)
    z_hat_0 = stats.norm.ppf(bias_factor)
    
    #estimate of coefficient fit theta-variance relationship as linear
    a_hat = (np.sum((jack_r2c_dot - jack_r2c)**3)/
             (6.*(np.sum((jack_r2c_dot - jack_r2c)**2))**(3/2.)))
    
    
    if z_hat_0==np.inf:
        alpha_1 = 1
        alpha_2 = 1
    elif z_hat_0==-np.inf:
        alpha_1 = 0
        alpha_2 = 0
    else:
        alpha_1 = stats.norm.cdf(z_hat_0 + 
                                 (z_alpha + z_hat_0)/(1 - a_hat*(z_alpha + z_hat_0))) 
        alpha_2 = stats.norm.cdf(z_hat_0 + 
                                 (z_1_alpha + z_hat_0)/(1 - a_hat*(z_1_alpha + z_hat_0)))
    
    #ci_low = np.nanpercentile(r2c_pbs, alpha_1*100, interpolation='lower')
    #ci_high = np.nanpercentile(r2c_pbs, alpha_2*100, interpolation='lower')  
    ci_low = np.nan
    ci_high = np.nan
    try:
        ci_low = percentile(r2c_pbs, alpha_1*100.)
        ci_high = percentile(r2c_pbs, alpha_2*100.)
    except:
        print('alpha_1=' + str(alpha_1))
        print('alpha_2='  + str(alpha_2))
        print('jack_r2c_dot='  + str(jack_r2c_dot))
        print('r2c_hat_obs='  + str(r2c_hat_obs))
        print('bias_factor='  + str(bias_factor))
        print('z_hat_0='  + str(z_hat_0))
        print('a_hat='  + str(a_hat))
        print('r2c_pbs='  + str(r2c_pbs))
        print('jack_r2c='  + str(r2c_pbs))

        
    return [ci_low, ci_high]

#%% set of parameters over which to test
m = 50
n =  4
n_exps = 5000
r2s = np.linspace(0, 1, 20)
trunc_sig2=[0.1, 1.5]
trunc_d2=[0.1, 1.5]
sig2 = np.random.uniform(trunc_sig2[0], trunc_sig2[1], size=n_exps);
d2 = np.random.uniform(trunc_d2[0], trunc_d2[1], size=n_exps);
alpha_targ = 0.2

#%% store all simulations ahead of time
yss = []
xs = []
for r2 in r2s:
    ys = []
    for a_d2, a_sig2 in zip(sig2, d2):
        theta = [r2, a_sig2, 1, a_d2*m, m, n] 
        x, y = pds_n2m_r2c(theta, 1, ddof=1)
        ys.append(y);
    yss.append(ys)
    xs.append(x.squeeze())
xs = np.array(xs)  
yss = np.array(yss)

x = xr.DataArray(xs, dims=['r2er', 'm'], 
                 coords=[list(r2s),] + [range(s) for s in xs.shape[1:]], name='x')
y = xr.DataArray(yss, dims=['r2er', 'exp', 'n', 'm'], 
                 coords=[list(r2s),] + [range(s) for s in yss.shape[1:]], name='y')

#%% run each ci method on the same data and store need to make xarray to save results.
print()
n_bs_samples = 5000
ciss = []
for r2 in tqdm(r2s):
    cis = []
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(get_pbs_bca_ci, args=(x.sel(r2er=r2).values, 
                         y.sel(r2er=r2, exp=exp).values, 
                         alpha_targ, n_bs_samples)) 
                            for exp in range(n_exps)]
    output = [p.get() for p in results]
    pool.close()   
    ciss.append([ar for ar in output])
ci_pbs_bca = np.array(ciss)

#%%
print('bayes')
n_splits = 100
n_r2c_sims = 5000
p_thresh = 0.01

cis = []
for r2 in tqdm((r2s)):
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(get_hyb_bayes_ci, args=(x.sel(r2er=r2).values, 
                                  y.sel(r2er=r2, exp=exp).values,  
                                n_r2c_sims, 
                                alpha_targ, 
                                p_thresh,
                                n_splits,
                                trunc_sig2, 
                                trunc_d2)) for exp in (range(n_exps))]

    output = [p.get() for p in results]
    pool.close()   

        
        
    cis.append([ar[:2] for ar in output])
ci_hyb_bayes = np.array(cis)

#%%
print('bs')
ciss = []
for r2 in tqdm(r2s):
    cis = []
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(get_npbs_ci, args=(x.sel(r2er=r2).values, 
                         y.sel(r2er=r2, exp=exp).values, 
                         alpha_targ, n_bs_samples)) 
                            for exp in range(n_exps)]
    output = [p.get() for p in results]
    pool.close()   
    ciss.append([ar for ar in output])
ci_npbs = np.array(ciss)

#%% parametric bootstap
print('npbs')
ciss = []
for r2 in tqdm(r2s):
    cis = []
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(get_pbs_ci, args=(x.sel(r2er=r2).values, 
                         y.sel(r2er=r2, exp=exp).values, 
                         alpha_targ, n_bs_samples)) 
                            for exp in range(n_exps)]
    output = [p.get() for p in results]
    pool.close()   
    ciss.append([ar for ar in output])
ci_pbs = np.array(ciss)

#%%

ci = xr.DataArray([ci_npbs, ci_pbs, ci_pbs_bca, ci_hyb_bayes],
             dims = ['meth', 'r2er', 'exp', 'ci'],
             coords =[['bs', 'pbs', 'pbs_bca', 'hbys'], r2s, list(range(n_exps)), ['ll', 'ul']],
             name='cis')

params = xr.DataArray([sig2, d2], dims=['p', 'exp'], 
                      coords=[['sig2', 'd2'], list(range(n_exps))], name='params')
params.attrs = {'m':m, 'n':n, 'alpha_targ':alpha_targ}


ds = xr.Dataset({ci.name:ci,params.name:params,x.name:x,y.name:y})

ds.to_netcdf('./ci_sim_data_m='+str(m)+'.nc')

#%%
#%%
'''
ci_pbs_bca_da =xr.DataArray(ci_pbs_bca,
             dims = [ 'r2er', 'exp', 'ci'],
             coords = [r2s, list(range(n_exps)), ['ll', 'ul']],
             name='cis')



import matplotlib.pyplot as plt
cis = ci_pbs_bca_da
r2ers = cis.coords['r2er']

cis_vals = cis.values
#cis_vals[cis_vals>1] = 1
#cis_vals[cis_vals<0] = 0

cis[...] = cis_vals

cis = cis[...,:,:]

in_ci = ((cis.sel(ci='ll')<=r2ers)*(cis.sel(ci='ul')>=r2ers)*(cis.sel(ci='ll')!=1)*(cis.sel(ci='ll')!=1)*(cis.sel(ci='ul')!=0))
'''
#%%

'''
from scipy import stats
import matplotlib.pyplot as plt

alpha = 0.1
ind = 3
a_y = y[0, ind].values
a_x = x[0].values
orig_r2c = rc.r2c_n2m(a_x.squeeze(), a_y)[0].squeeze()
#need percentiles alpha and 1-alpha
z_alpha = stats.norm.ppf(alpha)
z_1_alpha = stats.norm.ppf(1-alpha)
m = a_y.shape[-1]
n = a_y.shape[-2]
#get bootstrap factors, or replace this with NPS
n_bs_samples = 2000
# y_obs = a_y
# y_bs = []
# for k in range(n_bs_samples):
#     _ = np.array([np.random.choice(y_obs, size=n, replace=True) 
#                   for y_obs in a_y.T]).T
#     y_bs.append(_)
# y_bs = np.array(y_bs)
# r2c_bs = rc.r2c_n2m(a_x.squeeze(), y_bs)[0].squeeze()

r2c_hat_obs = rc.r2c_n2m(a_x.squeeze(), a_y)[0].squeeze()
hat_sig2 = s2_hat_obs_n2m(a_y)
hat_mu2y = mu2_hat_unbiased_obs_n2m(a_y)
theta = [r2c_hat_obs, hat_sig2, 1, hat_mu2y, m, n] 
x_new, y_new = pds_n2m_r2c(theta, n_bs_samples, ddof=1)

r2c_pbs = rc.r2c_n2m(x_new.squeeze(), y_new)[0].squeeze()

#jack knife
jack_r2c = []
for i in range(m):
    jack_a_y = np.array([a_y[:,j] for j in range(m) if i!=j]).T
    jack_a_x = np.array([a_x[j] for j in range(m) if i!=j])
    _ = rc.r2c_n2m(jack_a_x, jack_a_y)[0].squeeze()
    jack_r2c.append(_.squeeze())
    
jack_r2c = np.array(jack_r2c)
jack_r2c_dot = jack_r2c.mean()

#need bias correction factor
z_hat_0 = stats.norm.ppf(np.mean(r2c_pbs<orig_r2c))
#estimate of coefficient fitt theta-variance relationship as linear
a_hat = (np.sum((jack_r2c_dot - jack_r2c)**3)/
         (6.*(np.sum((jack_r2c_dot - jack_r2c)**2))**(3/2.)))


alpha_1 = stats.norm.cdf(z_hat_0 + 
                         (z_alpha + z_hat_0)/(1 - a_hat*(z_alpha + z_hat_0)))

alpha_2 = stats.norm.cdf(z_hat_0 + 
                         (z_1_alpha + z_hat_0)/(1 - a_hat*(z_1_alpha + z_hat_0)))

plt.hist(r2c_pbs);
plt.plot([orig_r2c, orig_r2c],[0,50])
ci_low = np.percentile(r2c_pbs, alpha_1*100)
ci_high = np.percentile(r2c_pbs, alpha_2*100)

plt.plot([ci_low, ci_low],[0,50])
plt.plot([ci_high, ci_high],[0,50])
#%%
def get_pbs_bca_ci(x,y, alpha_targ, n_bs_samples):
    #need percentiles alpha and 1-alpha
    z_alpha = stats.norm.ppf(alpha)
    z_1_alpha = stats.norm.ppf(1-alpha)
    m = y.shape[-1]
    n = y.shape[-2]
    #get bootstrap factors, or replace this with NPS
    n_bs_samples = 2000
    # y_obs = y
    # y_bs = []
    # for k in range(n_bs_samples):
    #     _ = np.array([np.random.choice(y_obs, size=n, replace=True) 
    #                   for y_obs in y.T]).T
    #     y_bs.append(_)
    # y_bs = np.array(y_bs)
    # r2c_bs = rc.r2c_n2m(a_x.squeeze(), y_bs)[0].squeeze()
    r2c_hat_obs = rc.r2c_n2m(x.squeeze(), y)[0].squeeze()
    hat_sig2 = s2_hat_obs_n2m(y)
    hat_mu2y = mu2_hat_unbiased_obs_n2m(y)
    theta = [r2c_hat_obs, hat_sig2, 1, hat_mu2y, m, n] 
    x_new, y_new = pds_n2m_r2c(theta, n_bs_samples, ddof=1)
    
    r2c_pbs = rc.r2c_n2m(x_new.squeeze(), y_new)[0].squeeze()
    
    #jack knife
    jack_r2c = []
    for i in range(m):
        jack_y = np.array([y[:,j] for j in range(m) if i!=j]).T
        jack_x = np.array([x[j] for j in range(m) if i!=j])
        _ = rc.r2c_n2m(jack_x, jack_y)[0].squeeze()
        jack_r2c.append(_.squeeze())
        
    jack_r2c = np.array(jack_r2c)
    jack_r2c_dot = jack_r2c.mean()
    
    #need bias correction factor
    z_hat_0 = stats.norm.ppf(np.mean(r2c_pbs<r2c_hat_obs))
    
    #estimate of coefficient fit theta-variance relationship as linear
    a_hat = (np.sum((jack_r2c_dot - jack_r2c)**3)/
             (6.*(np.sum((jack_r2c_dot - jack_r2c)**2))**(3/2.)))
    
    
    alpha_1 = stats.norm.cdf(z_hat_0 + 
                             (z_alpha + z_hat_0)/(1 - a_hat*(z_alpha + z_hat_0)))
    
    alpha_2 = stats.norm.cdf(z_hat_0 + 
                             (z_1_alpha + z_hat_0)/(1 - a_hat*(z_1_alpha + z_hat_0)))
    
    ci_low = np.percentile(r2c_pbs, alpha_1*100)
    ci_high = np.percentile(r2c_pbs, alpha_2*100)    
    return [ci_low, ci_high]
    
#get_pbs_ci(x, y, alpha_targ, n_pbs_samples=1000)


#%%%
#trying to center around estimate
b = stats.norm.ppf(np.mean(r2c_bs<=orig_r2c))

a = a_hat

ci_low = np.percentile(r2c_bs, 
                    100*stats.norm.cdf(
                        b-(z_alpha-b)/(1+a*(z_alpha-b))))

ci_high = np.percentile(r2c_bs, 
                    100*stats.norm.cdf(
                        b-(z_1_alpha-b)/(1+a*(z_1_alpha-b))))

print([ci_low, ci_high])



import matplotlib.pyplot as plt
plt.hist(r2c_bs)

#%%
alpha_2 = stats.norm.cdf(z_hat_0 + (z_hat_0+z_1_alpha)/
                                     (1-a_hat*(z_hat_0+z_1_alpha)))





#%%

# #%%
# in_cis = []
# for i, res in enumerate(ress):
#     cis = np.array([r[:2] for r  in res])
#     in_ci = (cis[:,0]<=r2s[i])*(cis[:,1]>=r2s[i])*(cis[:,0]!=cis[:,1])
#     in_cis.append(in_ci)
#     p = binom_test(np.sum(~in_ci), len(in_ci), p=alpha_targ)
#     plt.figure()
#     plt.title('Simulation of ' r'$r^2_{ER}$' 'CI \ntarget alpha=' + str(alpha_targ) + ', '+ 
#           'sim. alpha=' + str(np.round(np.mean(~in_ci), 2))+
#           ', p-val = ' + str(np.round(p, 2)) + '\n' +
#           r'$\sigma^2 \sim U[0.1,2], \ d^2 \sim U[0.1, 2]$')
    
#     plt.plot(cis[:, 0], c='r')
#     plt.plot(cis[:, 1], c='g')
#     plt.scatter(range(len(in_ci)), ~np.array(in_ci), s=5, color='k')
#     #plt.scatter(range(len(in_ci)), np.array(cis[:,0]==cis[:,1]))
#     plt.ylim(0,1.1)
#     plt.yticks(r2s)
#     plt.grid()

#     plt.legend(['upper ci', 'lower ci'][::-1])

#%%




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
'''


