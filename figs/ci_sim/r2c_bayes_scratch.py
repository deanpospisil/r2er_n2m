#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:23:16 2020

@author: dean
"""

from scipy.stats import ncx2
from scipy.stats import multivariate_normal
from scipy.stats import binom_test

import r2c_common as rc

import numpy as np
import matplotlib.pyplot as plt
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
def get_ci(x, y, n_r2c_eval=50, n_r2c_sims=500, r2c_range=[0,1], 
           alpha_targ=0.1, trunc_sig2=[0, np.inf], trunc_d2=[0, np.inf]):
    #get confidence intervals
    r2c_hat_obs = rc.r2c_n2m(x.squeeze(), y)[0]
    #r2c_hat_obs = trunc_ud(1, 0, r2c_hat_obs)
    trace, p = sample_post_s2_d2(y, n_samps=1000, 
                                 trunc_sig2=trunc_sig2, trunc_d2=trunc_d2)# get posterior dist of params
    sig2_post = trace[1]#trial-to-trial variance
    d2m_post = trace[0]*m#dynamic range
    sample_inds = np.random.choice(len(sig2_post), 
                                   size=n_r2c_sims, 
                                   replace=True)#randomly sample from post-dist
    
    #get distribution of r2er_hat
    res = np.zeros([n_r2c_eval, n_r2c_sims])
    r2cs =  np.linspace(r2c_range[0], r2c_range[1], n_r2c_eval)
    for i, r2c in (enumerate(r2cs)):
      for j in range(n_r2c_sims):
        k = sample_inds[j]
        theta = [r2c, sig2_post[k], d2m_post[k], d2m_post[k], m, n]
        x, y = pds_n2m_r2c(theta, 1, ddof=1)
        res[i,j] = rc.r2c_n2m(x.squeeze(), y)[0]
    
    alpha_obs = np.mean(res>r2c_hat_obs, 1)
    
    ll_dif = np.abs(alpha_obs-alpha_targ/2)
    ll_ind = np.argmin(ll_dif)
    if ll_dif[ll_ind]<alpha_targ:
        ll = r2cs[ll_ind]
    else:
        ll = 0
    
    ul_dif = np.abs(alpha_obs-(1-alpha_targ/2))
    ul_ind = np.argmin(ul_dif)
    if ul_dif[ul_ind]<alpha_targ:
        ul = r2cs[ul_ind]
    else:
        ul=1
    return ll, ul, alpha_obs, r2c_hat_obs, res, trace
   

# #def get_alpha(trace, r2c_hat_obs)

# theta = [1, 1, 1, 1, 40, 10]
# x, y = pds_n2m_r2c(theta, 1, ddof=1)
# hat_sig2 = y.var(0, ddof=1).mean(0)#central chisquared
# hat_d2 = y.mean(0).var(0, ddof=1)#non-central chisquared
# n_samps = 1000
# trace, ap = sample_post_s2_d2(y, n_samps, trunc_sig2=[0, 1], trunc_d2=[0, 1])
# plt.scatter(trace[0], trace[1], s=1)
# plt.ylim(0,2.1);
# plt.xlim(0,2.1);
# plt.title('posterior distribution \n'+ 
#           r'$\hat{d}^2=$' + str(np.round(hat_d2, 3).squeeze()) +
#           r', ${s}^2=$' + str(np.round(hat_sig2, 3).squeeze()))
# plt.xlabel(r'$\sigma^2|\hat{s}^2$')
# plt.ylabel(r'$d^2|\hat{d}^2$');
#%%




#%%

sig2 = 0.25
snr = 0.5
m = 40
n =  4
ncps = snr*m
mu2y = mu2x = ncps*sig2
r2c_range = [0, 1]
alpha_targ = 0.2
n_r2c_eval = 30
n_r2c_sims = 500
r2s = np.linspace(0, 1, 5)
n_exps = 1000

r2 = 1
theta = [r2, sig2, mu2x, mu2y, m, n] 
x, y = pds_n2m_r2c(theta, n_exps, ddof=1)# simulate experiment
r2c_hat_obs = np.squeeze(rc.r2c_n2m(x.squeeze(), y)[0])
plt.subplot(212)

plt.hist(r2c_hat_obs, cumulative=True, histtype='step', bins=10000, density=True, 
         range=[-1,2], color='b')
plt.xlabel(r'$\hat{r}^2_{ER}$')
#plt.title(np.round(np.median(r2c_hat_obs),4))

q = np.quantile(r2c_hat_obs, 0.9)
plt.plot([q,q], [0,1], c='b')


greater_inds = r2c_hat_obs>q
y_greater = y[greater_inds]
r2c_hat_obs_greater = r2c_hat_obs[greater_inds]

small_ind = np.argmin(r2c_hat_obs_greater)


ll, ul, alpha_obs, r2c_hat_obs_re, res_sim, trace =  get_ci(x, y_greater[small_ind], 
       n_r2c_eval=2, 
       n_r2c_sims=1000, 
       r2c_range=[0, 1], alpha_targ=0.1)

plt.hist(res_sim[-1], cumulative=True, histtype='step', bins=1000, density=True,
         range=[-1,2], color='r')
plt.xlim(0.5, 1.2);
plt.legend(['90% quantile', r'$\hat{r}^2_{ER}| \sigma^2=$' + str(sig2) + 
            r'$, d^2=$' + str(mu2x) + r'$, \ r^2_{ER}=1$', 
            r'$\hat{r}^2_{ER}| s^2, \hat{d}^2, r^2_{ER}=1$' ],
           fontsize=10)


sig2_post = trace[1]#trial-to-trial variance
d2m_post = trace[0]*m#dynamic range

plt.subplot(211)
plt.scatter(sig2_post[::10], d2m_post[::10])
plt.xlabel(r'$\sigma^2$')
plt.ylabel(r'$d^2$')
hat_sig2 = y_greater[small_ind].var(0, ddof=1).mean(0)#central chisquared
hat_d2 = y_greater[small_ind].mean(0).var(0, ddof=1)*m#
plt.title(r'$s^2=$' +str(np.round(hat_sig2, 2)) + ', '
          r'$\hat{d}^2=$' +str(np.round(hat_d2, 2)) )
plt.tight_layout()
#%%

n_exps = 50
n_r2c_eval = 50
n_r2c_sims = 200
res = []
for i, r2 in tqdm(enumerate(r2s)):
    theta = [r2, sig2, mu2x, mu2y, m, n] 
    x, y = pds_n2m_r2c(theta, n_exps, ddof=1)# simulate experiment
    for j in range(n_exps):
        ll, ul, alpha_obs, r2c_hat_obs, _, trace = get_ci(x, y[j], 
                                           n_r2c_eval=n_r2c_eval, 
                                           n_r2c_sims=n_r2c_sims, 
                                           r2c_range=r2c_range, 
                                           alpha_targ=alpha_targ)
        res.append([ll, ul, r2c_hat_obs, alpha_obs])
        
        
    #%%
plt.subplot(121)
plt.ylim(-0.1, 1.1);

for i in range(len(r2s)):
    in_ci = (res[0, i]<r2s[i])*(res[1, i]>r2s[i])
    print(np.sum(in_ci)/len(res[0, i]))
    plt.plot(res[0, i], c='r')
    plt.plot(res[1, i], c='g')
    
plt.yticks(r2s)
plt.grid()

plt.xlabel('simulation')
plt.ylabel(r'$r^2_{ER}$')

plt.legend(['upper ci', 'lower ci'])
plt.savefig('r2.pdf')

plt.title('m='+str(m) +', n='+ str(n)+r', $\sigma^2=$'+ str(sig2) +', SNR='+ str(snr) )

plt.subplot(122)
alpha_hats = []
ps = []
for i in range(len(r2s)):
    in_ci = (res[0, i]<=r2s[i])*(res[1, i]>=r2s[i])
    alpha_hats.append((np.sum(in_ci)/len(in_ci)))
    p = binom_test(np.sum(in_ci), len(in_ci), p=1-alpha_targ)
    ps.append(p)
alpha_hats = np.array(alpha_hats)
plt.plot(r2s, 1-alpha_hats, '-o')
#plt.plot(r2s, ps, '-o')

plt.xlabel(r'$r^2_{ER}$')
plt.ylabel(r'$\hat{\alpha}$')
plt.plot([0,1], [alpha_targ, alpha_targ])

plt.tight_layout()

#%% full bayesian sim
n_r2c_eval = 50
n_r2c_sims = 300
r2c_range = [0,1]
alpha_targ = 0.2
n_exps = 300
sig2 = np.random.uniform(0.1, 2, size=n_exps);
d2 = np.random.uniform(0.1, 2, size=n_exps);
r2 = 0
m = 30
n = 4
ys = []
for a_d2, a_sig2 in zip(sig2, d2):
    theta = [r2, a_sig2, 1, a_d2*m, m, n] 
    x, y = pds_n2m_r2c(theta, 1, ddof=1)
    ys.append(y)
ys = np.array(ys)
print(ys.shape)
ress = []
for y in tqdm(ys):
        ll, ul, alpha_obs, r2c_hat_obs, _, trace = get_ci(x, y, 
                                       n_r2c_eval=n_r2c_eval, 
                                       n_r2c_sims=n_r2c_sims, 
                                       r2c_range=r2c_range, 
                                       alpha_targ=alpha_targ,
                                       trunc_sig2=[0.1, 2], 
                                       trunc_d2=[0.1, 2])
        ress.append([ll, ul, alpha_obs, r2c_hat_obs, _, trace])


#%%   
alphas = np.array([r[2][0] for r in ress])

cis = np.array([r[:2] for r in ress])
#cis[0, 0] = r2s[-4]
in_ci = (cis[:, 0]<=0)*(cis[:, 1]>=0)*(0!=cis[:, 1])
plt.plot(cis[:, 0], c='r')
plt.plot(cis[:, 1], c='g')
plt.ylim(0,1.1)

p_hat = (alphas>1-alpha_targ/2) + (alphas<alpha_targ/2)
p_hat = in_ci
p = binom_test(np.sum(p_hat), len(p_hat), p=alpha_targ)
print(p)

plt.title('Simulation of ' r'$r^2_{ER}$' 'CI \ntarget alpha=' + str(alpha_targ) + ', '+ 
          'sim. alpha=' + str(np.round(np.mean(p_hat), 2))+
          ', p-val = ' + str(np.round(p, 2)) + '\n' +
          r'$\sigma^2 \sim U[0.1,2], \ d^2 \sim U[0.1, 2]$')

plt.xlabel('simulation')
plt.ylabel(r'$r^2_{ER}$')

plt.legend(['upper ci', 'lower ci'][::-1])
plt.savefig('./figs/ci_sim/r2_er=1_ci_simulation.pdf')
#ex_ind = np.array(list(range(len(alphas))))[alphas<alpha_targ/2][1]
ex_ind = 5
#ll, ul, alpha_obs, r2c_hat_obs, res, trace = ress[ex_ind]


#%%
plt.title('Prior distribution on parameters \n'
          r'example $\sigma^2=$' + str(np.round(sig2[ex_ind],2)) + 
          r', $d^2=$' + str(np.round(d2[ex_ind],2)))
plt.scatter(sig2, d2)
plt.scatter(sig2[ex_ind], d2[ex_ind]);
plt.xlabel(r'$\sigma^2$')
plt.ylabel(r'$d^2$');
plt.legend(['all', 'example'])
plt.savefig('./figs/ci_sim/prior_dist.pdf')

#%%
y = ys[ex_ind]
s2 = y.var(0, ddof=1).mean()#central chisquared
hat_d2 = y.mean(0).var(ddof=1)#non-central chisquared
s = np.linspace(0, 2*np.pi, int(m))
plt.errorbar(s, y.mean(0), yerr=y.std(0))
plt.plot(s,x)
r2c_hat_obs, r2 = rc.r2c_n2m(x.squeeze(), y)
plt.title(r'$\hat{r}^2_{ER}=$' + str(np.round(r2c_hat_obs,3).squeeze()) +
          r', $r^2=$' + str(np.round(r2,3).squeeze()) +
          r', $\hat{d}^2=$' + str(np.round(hat_d2,3).squeeze()) +
          r', ${s}^2=$' + str(np.round(s2,3).squeeze()));
plt.legend(['Y (data w/ std)', 'X (model)'][::-1])
plt.savefig('./figs/ci_sim/x_y_sim.pdf')


#%%
plt.title('Confidence interval method')
ll, ul, alpha_obs, _, res, trace = ress[ex_ind]
r2s = np.linspace(0, 1, n_r2c_eval)
plt.scatter(r2s[ll==r2s], alpha_obs[ll==r2s], c='red')
plt.scatter(r2s[-1], alpha_obs[-1], facecolor='none', edgecolor='green')


plt.plot(r2s, alpha_obs, '-')

plt.hist(res[ll==r2s].squeeze(), histtype='step', cumulative=True, density=True, 
         bins=10000, color='red')
plt.hist(res[-1], histtype='step', cumulative=True, density=True, 
         bins=10000, color='green')
plt.plot([r2c_hat_obs.squeeze(), r2c_hat_obs.squeeze()], [0, 1])

plt.legend([r'$P[\hat{r}^2_{ER} \geq x | {r}^2_{ER}]$',
            r'$\hat{r}^2_{ER}$', 'CDF lower ci', 'CDF upper ci', 
            'lower ci', 'upper ci'])
plt.xlabel(r'$r^2_{ER}$')
plt.xlim(0.,1.2)
plt.ylim(0,1);plt.yticks(np.linspace(0,1,11));plt.grid()
plt.savefig('./figs/ci_sim/conf_int_method.pdf')


#%%

plt.scatter(trace[1], trace[0], s=1)
plt.ylim(0,2.1);
plt.xlim(0,2.1);
plt.title('posterior distribution \n'+ 
          r'$\hat{d}^2=$' + str(np.round(hat_d2,3).squeeze()) +
          r', ${s}^2=$' + str(np.round(s2, 3).squeeze()))
plt.xlabel(r'$\sigma^2|\hat{s}^2$')
plt.ylabel(r'$d^2|\hat{d}^2$');
plt.savefig('./figs/ci_sim/posterior_dist.pdf')

#%%
'''
plt.acorr(trace[0]-trace[0].mean(), maxlags=200)
n_exps = 2000
r2 = 0.5
sig2 = 0.25
snr = 2
m = 30
n =  5
ncps = snr*m
mu2y = mu2x = ncps*sig2

theta = [r2, sig2, mu2x, mu2y, m, n] 
x, y, yu = pds_n2m_r2c(theta, n_exps, ddof=1)

s2 = y.var(-2, ddof=1).mean(-1)#central chisquared
d2 = y.mean(-2).var(-1,ddof=1)#non-central chisquared


df = m*(n-1)
nc = 0
scale = sig2/(m*(n-1))
x = np.linspace(0, np.max(s2))
fs2 = ncx2.pdf(x, df, nc, scale=scale)
plt.plot(x, fs2)
plt.hist(s2, density=True)
vs2 = ncx2.stats(df, nc, loc=0, scale=scale, moments='v')

plt.figure()
df = (m-1)
nc = (mu2y)/(sig2/n)
scale = sig2/(n*(m-1))
x = np.linspace(0, np.max(d2))
fd2 = ncx2.pdf(x, df, nc, scale=scale)

plt.plot(x, fd2)
plt.hist(d2, density=True)



hat_sig2 = s2[0]
hat_d2 = d2[0]

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


accept = np.zeros(5000)
trace = []
for i in tqdm(range(5000)):
    sig2_cand, d2_cand = np.random.normal(loc=[sig2_curr, d2_curr], 
                                          scale=[vs2**0.5, vd2**0.5])
    
    scale = sig2_cand/(n*(m-1))
    fd2_cand = ncx2._pdf(hat_d2*(scale**-1), 
                        df=m-1, 
                        nc=(d2_cand*m)/(sig2_cand/n))*(scale**-1)
    
    scale = sig2_curr/(n*(m-1))
    fd2_curr = ncx2._pdf(hat_d2*(scale**-1), 
                        df=m-1, 
                        nc=(d2_curr*m)/(sig2_curr/n))*(scale**-1)
    
    
    scale = sig2_cand/(m*(n-1))
    fs2_cand = ncx2._pdf(hat_sig2*(scale**-1), df=m*(n-1), nc=0)*(scale**-1)
    scale = sig2_curr/(m*(n-1))
    fs2_curr = ncx2._pdf(hat_sig2*(scale**-1), df=m*(n-1), nc=0)*(scale**-1)
    
    a1 = (fs2_cand*fd2_cand)/(fs2_curr*fd2_curr)
    
    
    # g_cand = multivariate_normal.pdf([sig2_curr, d2_curr], 
    #                                  mean=[sig2_cand, d2_cand], 
    #                                  cov=[[vs2, 0],
    #                                       [0, vd2]])
    
    # g_curr = multivariate_normal.pdf([sig2_cand, d2_cand], 
    #                                  mean=[sig2_curr, d2_curr], 
    #                                  cov=[[vs2, 0],
    #                                       [0, vd2]])
    #a2 = g_cand/g_curr
    a = a1#*a2
    if a>1:
        sig2_curr = sig2_cand
        d2_curr = d2_cand
        accept[i]=1
    else:
        u = np.random.uniform(0, 1)
        if u < a:
            accept[i]=1
            sig2_curr = sig2_cand
            d2_curr = d2_cand
                
    
    trace.append([d2_curr,sig2_curr])
    
plt.plot(trace)

plt.figure()
trace = np.array(trace)
plt.scatter(trace[10:,0], trace[10:,1])
plt.title([hat_sig2, hat_d2])

print(np.mean(accept))
plt.figure()
plt.acorr(trace[:,0]-trace[:,0].mean(), maxlags=200)[1]

'''