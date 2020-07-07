#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:06:23 2020

@author: dean
"""


import os
cwd = os.getcwd()
os.chdir('../../')
import xarray as xr
import  matplotlib.pyplot as plt
import numpy as np

import pandas as pd


def ravel_cor(cor):
    cor[np.triu_indices(cor.shape[0])] = np.nan
    cor = cor.ravel()
    ind = ~np.isnan(cor)
    cor = np.array(cor[ind])
    return cor

def ravel_cor_inds(nunits):
    cor = np.zeros((nunits,nunits))
    cor[np.triu_indices(nunits)] = np.nan
    cor = cor.ravel()
    ind = ~np.isnan(cor) 
    xy = np.array(np.meshgrid(np.arange(nunits),np.arange(nunits)))
    a = xy[0].ravel()[ind]
    b = xy[1].ravel()[ind]
    
    return a,b

def sig_cor(r):
    r_temp = r.copy(deep=True)
    mt = r_temp.mean('trial', skipna=True)
    s_cor = np.corrcoef(mt.T)**2
    s_cor = ravel_cor(s_cor)    

    return s_cor

def sig_cor_split(r):
    r_temp = r.copy(deep=True)
    
    nunits = len(r.coords['unit'])
    r1 = r_temp.isel(trial=slice(0,-1,2)).mean('trial')
    r2 = r_temp.isel(trial=slice(1,-1,2)).mean('trial')
    #s_cor_split
    cor  = np.corrcoef(r1.T, r2.T)**2
    s_cor_split = (cor[:nunits, nunits:] + cor[nunits:, :nunits])/2

    s_cor_split = ravel_cor(s_cor_split)    

    return s_cor_split

#%%
load_dir = './data/mt_dotd/'
fns = os.listdir(load_dir)
fns = [fn for fn in fns if '.nc' in fn ]
mt = [xr.open_dataarray(load_dir+fn).load() for  fn in fns]
fns = [fn.split('.')[0] for fn in fns]

os.chdir(cwd)

mt = xr.concat(mt, 'rec')
mt.coords['rec'] = range(len(fns))
mt.coords['nms'] = ('rec', [fn[:-1] for fn in fns])


mt_s = mt.sel(t=slice(0,2), unit=[0,1])
s = mt_s.sum('t', min_count=1)
s = s**0.5    
ind = ((~s.isnull()).sum('trial_tot')>9).prod('dir').prod('unit').values
s = s[(ind==1)] 
s = s.dropna('rec', how='all').dropna('trial_tot')
nms = s.coords['nms']

ss = xr.concat([s[:, 0], s[:, 1]], 'rec')
ss.coords['rec'] = range(len(ss.coords['rec']))
#ss.to_netcdf('./mt_sqrt_spkcnt.nc')
#mt = xr.open_dataarray('./mt_sqrt_spkcnt.nc')
mt = ss

#%%
#load np allen  grating
import os
load_dir = '../data/responses/allen/np/grating/'
load_dir = '/Users/dean/Desktop/code/science/modules/r2er_n2m/data/allen/np/np/grating/'
fns = os.listdir(load_dir)
fns = [fn for fn in fns if '.nc' in fn ]
visp_np = [xr.open_dataset(load_dir+fn).load() for  fn in fns]
visp_np = [visp_ for visp_ in visp_np if len(visp_)>0]
n_recs = len(visp_np)

for i, visp_ in enumerate(visp_np):
    
    visp_ = visp_['spike_counts']
    area = visp_.coords['area']
    visp_ = visp_[area=='VISp']
    visp_.coords['unit_id'] = range(len(visp_.coords['unit_id']))
    visp_np[i] = visp_
    

visp_np = xr.concat(visp_np, 'rec')
visp_np.coords['rec'] = range(n_recs)


#%%

#load np allen natural scene
load_dir = '/loc6tb/data/responses/np_allen/natural_scene/'
load_dir = '/Users/dean/Desktop/code/science/modules/r2er_n2m/data/allen/np/np/natural_scene/'
fns = os.listdir(load_dir)
fns = [fn for fn in fns if '.nc' in fn ]
visp_np_nat = [xr.open_dataset(load_dir+fn).load() for  fn in fns]
visp_np_nat = [visp_ for visp_ in visp_np_nat if len(visp_)>0]
n_recs = len(visp_np_nat)
#visp = xr.concat(visp, 'rec')

for i, visp_ in enumerate(visp_np_nat):
    
    visp_ = visp_['spike_counts']
    area = visp_.coords['area']
    visp_ = visp_[area=='VISp']
    visp_.coords['unit_id'] = range(len(visp_.coords['unit_id']))
    visp_np_nat[i] = visp_
    

visp_np_nat = xr.concat(visp_np_nat, 'rec')
visp_np_nat.coords['rec'] = range(n_recs)


visp_np_nat_s = visp_np_nat**0.5
u = (visp_np_nat_s.stack(u=('unit_id', 'rec')).dropna('u', how='all'))
u = u.dropna('trial', how='any')
u = u.rename({'u':'unit', 'frame':'stim'})
u = u.transpose('unit', 'trial', 'stim')
visp_np_nat_s = u

ns = (~visp_np_nat_s.isnull()).sum('trial', skipna=True).mean(('stim'), skipna=True)
m = len(visp_np_nat_s.coords['stim'])
v = visp_np_nat_s.var('trial', skipna=True).mean('stim', skipna=True)
sig = visp_np_nat_s.mean('trial', skipna=True).var(('stim'), ddof=0, skipna=True)
sig = sig - (m/(m-1))*v/ns
snr_visp_np_nat = sig/v
#snr_visp_np = snr_visp_np.stack(c=('rec', 'unit_id')).dropna('stim')
#%%
#load ca allen 
#from allensdk.core.brain_observatory_cache import BrainObservatoryCache
# top_dir = '/loc6tb/data/responses/ophys_allen/boc/'
# boc = BrainObservatoryCache(cache=True, 
#                             manifest_file='/loc6tb/data/responses/ophys_allen/boc/brain_observatory_manifest.json')
# lines = boc.get_all_cre_lines()
# depths = boc.get_all_imaging_depths()
# structs = boc.get_all_targeted_structures()
# stim = boc.get_all_stimuli()

# line = lines[0]
# depth = depths[0]
# struct = structs[0]

# exps = boc.get_ophys_experiments(
#                   targeted_structures=['VISp',],
#                   stimuli=['static_gratings'],
#                   require_eye_tracking=False)


load_dir = '/loc6tb/data/responses/ophys_allen/static_stim_resp/grating/'
load_dir = '/Users/dean/Desktop/code/science/modules/r2er_n2m/data/allen/ca/data/grating/'

#ids = [exp['id'] for exp in exps]
ids = [int(nm.split('.')[0]) for nm in os.listdir(load_dir) if '.nc' in nm]

visp_ca = [xr.open_dataset(load_dir + str(int(an_id)) + '.nc')['__xarray_dataarray_variable__']
           for an_id in ids]

load_dir = '/Users/dean/Desktop/code/science/modules/r2er_n2m/data/allen/ca/data/natural_scene/'
visp_ca_nat = [xr.open_dataset(load_dir + str(int(an_id)) + '.nc')['__xarray_dataarray_variable__']
               for an_id in ids]

#exps = boc.get_ophys_experiments(
#                  targeted_structures=['VISp',],
#                  stimuli=['static_gratings'],
#                  require_eye_tracking=False)
#
#ids = [exp['id']for exp in exps]
#
#load_dir = '/loc6tb/data/responses/ophys_allen/static_stim_resp/grating/'
#visp_ca_nat = [xr.open_dataset(load_dir + str(int(an_id)) + '.nc')['__xarray_dataarray_variable__']
# for an_id in ids]


#%% load APC
load_dir = '/Users/dean/Desktop/code/science/modules/r2er_n2m/data/apc370_with_trials.nc'
v4 = xr.open_dataset(load_dir)['resp']
try:
    v4 = v4.rename({'trials':'trial'})
    v4 = v4.rename({'shapes':'stim'})
except:
    print('')
v4.coords['unit'] = range(109)
ys = []
for cell in range(109):
    y = v4.sel(unit=cell).dropna('trial')
    if (y.shape[1])==4:
        ys.append(y)
v4_apc = xr.concat(ys, 'unit')

#%% load FO
v4_fo = xr.open_dataset('/Users/dean/Desktop/code/science/modules/r2er_n2m/data/fo.nc')
v4_fo = v4_fo['__xarray_dataarray_variable__']*.3
v4_fo_s = v4_fo**0.5

#%%
labels = ['dot', 'f', 'o', 
          'apc', 'aNP', 
          'aCA'] 
params = pd.DataFrame(np.zeros((len(labels), 2)), 
                      index=labels, columns=['n', 'm'])
#%%

# will need table of m, n, s^2, and var mean_resp
#wyeth v1

v = mt.var('trial_tot', skipna=True).mean('dir')
sig = mt.mean('trial_tot', skipna=True).var('dir', ddof=0)

ns = (~s.isnull()).sum('trial_tot').mean('dir')
m = len(mt_s.coords['dir'])
snr_mt = sig - (m/(m-1))*v/ns
snr_mt = sig/v

params.loc['dot'] = [ns.mean().values, m]

#%%
#fo v4

v4_f_s = v4_fo.sel(fo=0)**0.5
ns = (~v4_f_s.isnull()).sum('trial', skipna=True).mean(('stim'), skipna=True)
m = len(v4_f_s.coords['stim'])
v = v4_f_s.var('trial', skipna=True).mean('stim')
sig = v4_f_s.mean('trial', skipna=True).var('stim')
sig = sig - (m/(m-1))*v/ns
snr_v4_f = sig/v
params.loc['f'] = [ns.mean().values, m]


v4_o_s = v4_fo.sel(fo=1)**0.5
ns = (~v4_o_s.isnull()).sum('trial', skipna=True).mean(('stim'), skipna=True)
m = len(v4_o_s.coords['stim'])
v = v4_o_s.var('trial', skipna=True).mean('stim')
sig = v4_o_s.mean('trial', skipna=True).var('stim')
sig = sig - (m/(m-1))*v/ns
snr_v4_o = sig/v
params.loc['o'] = [ns.mean().values, m]





#%%
# apc v4
v4_apc_s = v4_apc**0.5
ns = (~v4_apc_s.isnull()).sum('trial', skipna=True).mean(('stim'), skipna=True)
m = len(v4_apc_s.coords['stim'])
v = v4_apc_s.var('trial', skipna=True).mean('stim')
sig = v4_apc_s.mean('trial', skipna=True).var(('stim'), ddof=0)
sig = sig - (m/(m-1))*v/ns
snr_v4_apc = sig/v
params.loc['apc'] = [ns.mean().values, m]

#%%
visp_np_s = visp_np**0.5

ns = (~visp_np_s.isnull()).sum('trial', skipna=True).mean(('c'), skipna=True)
m = len(visp_np_s.coords['c'])
v = visp_np_s.var('trial', skipna=True).mean('c', skipna=True)
sig = visp_np_s.mean('trial', skipna=True).var(('c'), ddof=0, skipna=True)
sig = sig - (m/(m-1))*v/ns
snr_visp_np = sig/v
snr_visp_np = snr_visp_np.stack(c=('rec', 'unit_id')).dropna('c')

params.loc['aNP'] = [ns.mean().values, m]

#%%
snr_visp_ca_ss=[]
nss=[]
ms = []
for visp_ca_s in visp_ca:

    visp_ca_s = visp_ca_s.dropna('trial')
    visp_ca_s = visp_ca_s/visp_ca_s.std('trial', skipna=True).mean('stim', skipna=True)
    ns = len(visp_ca_s.coords['trial'])
    m = len(visp_ca_s.coords['stim'])
    nss.append(ns)
    ms.append(m)
    v = visp_ca_s.var('trial', skipna=True).mean('stim', skipna=True)
    sig = visp_ca_s.mean('trial', skipna=True).var(('stim'), ddof=0, skipna=True)
    sig = sig - (m/(m-1))*v/ns
    snr_visp_ca_s = sig/v
    
    snr_visp_ca_ss.append(snr_visp_ca_s)
snr_visp_ca_ss = np.concatenate(snr_visp_ca_ss)
params.loc['aCA'] = [np.mean(nss), np.mean(ms)]

#%%
snr_visp_ca_ss_nat=[]
nss=[]
ms = []
for visp_ca_s in visp_ca_nat:

    visp_ca_s = visp_ca_s.dropna('trial')
    visp_ca_s = visp_ca_s/visp_ca_s.std('trial', skipna=True).mean('stim', skipna=True)
    ns = len(visp_ca_s.coords['trial'])
    m = len(visp_ca_s.coords['stim'])
    nss.append(ns)
    ms.append(m)
    v = visp_ca_s.var('trial', skipna=True).mean('stim', skipna=True)
    sig = visp_ca_s.mean('trial', skipna=True).var(('stim'), ddof=0, skipna=True)
    sig = sig - (m/(m-1))*v/ns
    snr_visp_ca_s = sig/v
    
    snr_visp_ca_ss_nat.append(snr_visp_ca_s)
snr_visp_ca_ss_nat = np.concatenate(snr_visp_ca_ss_nat)

#%%
import matplotlib as mpl
def fix_hist_step_vertical_line_at_end(ax):
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, mpl.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])
    
labels = ['MT dot motion 2s ',
          'V4 fill shapes 0.3s (2019)', 
          'V4 outline shapes 0.3s', 
          'V4 fill shapes 0.5s (2001)',
          'VISp gratings Allen NP 0.25s',
          'VISp natural Allen NP 0.25s', 
          'VISp gratings Allen Ca img.', 
          'VISp natural Allen Ca img.'] 
snrs = [snr_mt,  
        snr_v4_f,  
        snr_v4_o, 
        snr_v4_apc, 
        snr_visp_np, 
        snr_visp_np_nat, 
        snr_visp_ca_ss, 
        snr_visp_ca_ss_nat]

plt.figure(figsize=(8,4))
snr_t = np.array([2, 0.3, 0.3, 0.5, 0.25, 0.25, 1, 1])

for i in range(2): 
    plt.subplot(1,2,i+1)
    if i == 1:
        snr_t = np.array([2, 0.3, 0.3, 0.5, 0.25, 0.25, 1, 1])
    else:
        snr_t[...] = 1
    snrs_t = [snrs[i]/snr_t[i] for i in range(len(labels))]
    for t, snr in enumerate(snrs_t):
        ax = plt.gca()
        cnt, edges = np.histogram(snr, bins=10000, density=1)
        # plot the data as a step plot.  note that edges has an extra right edge.
        ax.step(edges[:-1], cnt.cumsum()/cnt.sum(), lw=2)

        #fix_hist_step_vertical_line_at_end(ax)
    plt.semilogx()    

    plt.yticks(np.linspace(0,1,5))
    plt.xticks(np.logspace(-3, 2, 6))
    plt.xlim(1e-3,1e2)
    plt.ylim(0,1.1)
    plt.grid()
    if i==0:
        plt.xlabel(r'$\widehat{SNR}$')
        plt.ylabel('Cumulative fraction units')
        plt.title('SNR of original experiment')
    else:
        plt.gca().set_yticklabels([])
        plt.title('SNR per second')

plt.legend(labels, loc=(1,0), fontsize=8)
plt.tight_layout()
plt.savefig('./snr_dist_plots.pdf')
#%%
'''
for snr in snrs:
    print(np.median(snr))
    print(len(snr))
    
#%%
u = (visp_np_s.stack(u=('unit_id', 'rec')).dropna('u', how='all'))
u = u.dropna('trial', how='any')
u=u.rename({'u':'unit', 'c':'stim'})
visp_ca_m = []
for i in range(len(visp_ca)):
    visp_ca_m.append(visp_ca[i].reset_index('stim'))
    visp_ca_m[i].coords['stim'] = range(len(visp_ca_m[i].coords['stim']))

#%%
visp_ca_m = xr.concat(visp_ca_m[:], 'unit')
#%%
unit_info = pd.DataFrame(np.zeros((len(visp_ca_m.coords['unit']),3)))
depth = []
line = []
exp_id = []
for i in range(len(exps)):
    depth += [exps[i]['imaging_depth'],]*len(visp_ca[i].coords['unit'])
    exp_id += [exps[i]['id'],]*len(visp_ca[i].coords['unit'])
    line += [exps[i]['cre_line'],]*len(visp_ca[i].coords['unit'])
    
#%%
sig_cor_np = sig_cor_split(u)
asnr_np = (u.mean('trial').var('stim')/u.var('trial').mean('stim')).values
asnr_np = (asnr_np[:,np.newaxis]*asnr_np[np.newaxis])
asnr_np = ravel_cor(asnr_np)
#%%

u = visp_ca_m.dropna('trial', how='any')
#u[..., 10580] = np.nan # one cell with high SNR
u = u.dropna('unit', how='all')
u = u/u.std('trial')
u = u.transpose('trial', 'stim', 'unit')

sig_cor_ca = sig_cor_split(u)
asnr_ca = (u.mean('trial').var('stim')/u.var('trial').mean('stim')).values
asnr_ca = (asnr_ca[:,np.newaxis]*asnr_ca[np.newaxis])
asnr_ca = ravel_cor(asnr_ca)
#%%
plt.scatter(asnr_np[::100], sig_cor_np[::100], s=1);
plt.semilogx()
plt.xlim(0.0001,10)
plt.xlabel('SNR');plt.ylabel('$r_{SC}$');

plt.scatter(asnr_ca[::1000], sig_cor_ca[::1000], s=1);
plt.semilogx()
plt.xlim(0.00001,10)
plt.xlabel('SNR');plt.ylabel('$r_{SC}$');
plt.ylim(0,1)
#%%
snr = (u.mean('trial').var('stim')/u.var('trial').mean('stim')).values
p = pd.DataFrame(snr, index=line).reset_index()
p.groupby('index').max()
ind = snr.argsort()[::-1][0]

m = u[..., ind].mean('trial')
sd = u[..., ind].std('trial')
plt.errorbar(range(len(sd)),y=m,yerr=sd)


#%%
from scipy.stats import ttest_ind

ttest_ind(sig_cor_ca, sig_cor_np, equal_var=False)

print(sig_cor_ca.mean())
print(sig_cor_np.mean())

print(sig_cor_ca.std())
print(sig_cor_np.std())


#%%
u = (visp_np_s.stack(u=('unit_id', 'rec')).dropna('u', how='all'))
u = u.dropna('trial', how='any')
u = u.rename({'u':'unit', 'c':'stim'})

a,b = ravel_cor_inds(len(u.coords['unit']))

#%%
from r2c_common import r2c_n2n
r2cs = []
for a_, b_ in zip(a[::1000], b[::1000]):
    x, y = u[...,[a_,b_]].T.values
    r2cs.append(r2c_n2n(x[:,::2].T, y[:,1::2].T))

r2cs = np.array(r2cs).squeeze()

#%%

plt.plot(r2cs);plt.ylim(0,1)

#%%
r2cs_rec = []
for rec in visp_np_s:
    temp = rec.copy(deep=True)
    temp = temp.dropna('unit_id', how='all').dropna('trial')
    print(temp.shape)
    a,b = ravel_cor_inds(len(temp.coords['unit_id']))
    temp = temp.transpose('trial','c', 'unit_id')
    for a_, b_ in zip(a[::1], b[::1]):
        x, y = temp[...,[a_,b_]].T.values
        r2cs_rec.append(r2c_n2n(x[:,::2].T,y[:,1::2].T))
#%%
r2cs_rec = np.array(r2cs_rec).squeeze()
plt.scatter(r2cs_rec[:,0], r2cs_rec[:,1],s=1);
plt.scatter(r2cs[:, 0],  r2cs[:, 1], s=1)
plt.ylim(-1,1);
plt.xlim(-1,1)

#%%
plt.ylim(-1,1)
plt.scatter(asnr_np[::1000]**0.5,  r2cs[:, 0], s=1)
plt.scatter(asnr_np[::1000]**0.5,  r2cs[:, 1], s=1)
plt.xlim(0.001,10)
plt.semilogx()
plt.ylim(-1,1)

print(np.average(r2cs[:,0], weights=asnr_np[::1000]))
print(np.average(r2cs[:,1], weights=asnr_np[::1000]))

#%%

u = visp_ca_m.dropna('trial', how='any')
#u[..., 10580] = np.nan # one cell with high SNR
u = u.dropna('unit', how='all')
u = u/u.std('trial')
u = u.transpose('trial', 'stim', 'unit')

a,b = ravel_cor_inds(len(u.coords['unit']))

r2cs = []
for a_, b_ in zip(a[::10000], b[::10000]):
    x, y = u[...,[a_,b_]].T.values
    r2cs.append(r2c_n2n(x.T,y.T))

r2cs = np.array(r2cs).squeeze()

#%%
plt.ylim(-1,1)
plt.scatter(asnr_ca[::10000],  r2cs[:,0], s=1)
plt.scatter(asnr_ca[::10000],  r2cs[:,1], s=1)
plt.xlim(0.0001,1)
plt.semilogx()

print(np.average(r2cs[:,0], weights=asnr_ca[::10000]))
print(np.average(r2cs[:,1], weights=asnr_ca[::10000]))

#%%
#%% get alll
visp_np_s = visp_np**0.5
u = (visp_np_s.stack(u=('unit_id', 'rec')).dropna('u', how='all'))
u = u.dropna('trial', how='any')
u = u.rename({'u':'unit', 'c':'stim'})

a,b = ravel_cor_inds(len(u.coords['unit']))

r2cs = []
for a_, b_ in zip(a[::1], b[::1]):
    x, y = u[...,[a_,b_]].T.values
    r2cs.append(r2c_n2n(x[:,::2].T, y[:,1::2].T))

r2cs = np.array(r2cs).squeeze()



#%%
rec_nums = u.coords['rec']
rec_inds = [rec_nums[a].values, rec_nums[b].values]

rec_inds = np.array(rec_inds)
#%%
x, y = u[...,[a_,b_]].T.values
(r2c_n2n(x[:,::2].T, y[:,1::2].T))

#%%
visp_np_s = visp_np**0.5
u = (visp_np_s.stack(u=('unit_id', 'rec')).dropna('u', how='all'))
u = u.dropna('trial', how='any')
u = u.rename({'u':'unit', 'c':'stim'})
u = u.transpose('unit', 'trial', 'stim')
a,b = ravel_cor_inds(len(u.coords['unit']))

step = 100000
rs = []

for i,j in zip(list(range(0,len(a),step)), list(range(step,len(a),step))):
    print(i/len(a))
    rs.append(np.array((r2c_n2n(u[a[i:j],::2].values, u[b[i:j], 1::2].values))).squeeze())
rs.append(np.array((r2c_n2n(u[a[j:],::2].values, u[b[j:], 1::2].values))).squeeze())
rs = np.concatenate(rs,1)


r2cs_spl = r2c_n2n(u[:, ::2].values, u[:, 1::2].values)
r2cs_spl = np.concatenate(r2cs_spl,1).squeeze()


#%%
rec_nums = u.coords['rec']
rec_inds = [rec_nums[a].values, rec_nums[b].values]

rec_inds = np.array(rec_inds)


#%%
same = rec_inds[0]==rec_inds[1]
#%%
asnr_np = (u.mean('trial').var('stim')/u.var('trial').mean('stim')).values
asnr_np = (u.mean('trial').var('stim')).values
#asnr_np = (u.var('trial').mean('stim')).values

asnr_np = (asnr_np[:,np.newaxis]*asnr_np[np.newaxis])
asnr_np = ravel_cor(asnr_np)**0.5
snr_np_spl = (u.mean('trial').var('stim')/u.var('trial').mean('stim')).values

#%%

bs=100

    
def bs_med_var(x,bs):
    sd = np.std([np.median(x[np.random.choice(len(x), len(x))])
    for i in range(bs)])
    return sd

da_spl =xr.DataArray(r2cs_spl[np.argsort(snr_np_spl)], dims=('unit', 'm'),
             coords=[(np.sort(snr_np_spl)), range(2)])

n_perc_bins = 25
bins = np.percentile(snr_np_spl, np.linspace(20,100, n_perc_bins))

n_unit_per_bin = len(da_spl.coords['unit'])/n_perc_bins


#plt.scatter(da_spl[:,0].coords['unit'], da_spl[:,0], alpha=0.005)
#plt.scatter(da_spl[:,1].coords['unit'], da_spl[:,1], alpha=0.005)

avr2c = da_spl[:,0].groupby_bins('unit', bins).mean(skipna=True)
avr2c_sd = da_spl[:,0].groupby_bins('unit', bins).std(skipna=True)
g = [a[1].values for a in list(da_spl[:,0].groupby_bins('unit', bins))]
avr2c_sd = np.array([bs_med_var(x, bs) for x in g])


avr2 = da_spl[:,1].groupby_bins('unit', bins).mean(skipna=True)
avr2_sd = da_spl[:,1].groupby_bins('unit', bins).std(skipna=True)
g = [a[1].values for a in list(da_spl[:,1].groupby_bins('unit', bins))]
avr2_sd = np.array([bs_med_var(x, bs) for x in g])

bins_half = [(bins[1+i]+bins[i])/2 for i in range(len(bins)-1)]

plt.figure(figsize=(3,5))

plt.subplot(211)
plt.title('Allen Neuropixel Static Grating\nSplit-Half Correlation')
plt.errorbar(bins_half, avr2c, yerr=avr2c_sd*2)#/n_unit_per_bin**0.5)
plt.errorbar(bins_half, avr2, yerr=avr2_sd*2)#/n_unit_per_bin**0.5)

plt.semilogx()
plt.xlim(1e-2, 2.5);
plt.ylim(0,1.5)
plt.gca().set_xticklabels([])
plt.grid()
plt.legend([r'$\hat{r}^2$', r'$\hat{r}^2_{ER}$'][::-1])
plt.ylabel('$r^2$')

n_unit_per_bin = len(da_spl.coords['unit'])/n_perc_bins

bins = np.percentile(asnr_np, np.linspace(20,100, n_perc_bins))

da_rs = xr.DataArray(rs[:, np.argsort(asnr_np)[::1]], dims=('m', 'unit'),
             coords=[range(2), (np.sort(asnr_np)[::1])])

avr2c = da_rs[0].groupby_bins('unit', bins).median(skipna=True)
g = [a[1].values for a in list(da_rs[0].groupby_bins('unit', bins))]
avr2c_sd = np.array([bs_med_var(x, bs) for x in g])



avr2 = da_rs[1].groupby_bins('unit', bins).median(skipna=True)
g = [a[1].values for a in list(da_rs[1].groupby_bins('unit', bins))]
avr2_sd = np.array([bs_med_var(x, bs) for x in g])


plt.subplot(212)
plt.title('Signal Correlation')

bins_half = [(bins[1+i]+bins[i])/2 for i in range(len(bins)-1)]
plt.errorbar(bins_half, avr2c, yerr=avr2c_sd*2)
plt.errorbar(bins_half, avr2, yerr=avr2_sd*2)

plt.semilogx()
plt.xlim(1e-2, 2.5);


plt.ylim(0, .15)
plt.xlabel('Dynamic Range (Var(signal))')
plt.grid()
plt.annotate('\n# cells = ' + str(len(r2cs_spl[:,0]))
        + '\n# pairs = ' + str(len(da_rs[0])), (0.012, 0.1)
             )

plt.tight_layout()
plt.savefig('./figs/snr_r2c_plot.pdf')
#%%
def r2c_n2n_sig(x,y,sig2_hat, n=1):
    """approximately unbiased estimator of R^2 between the expected values. 
        of the columns of x and y. Assumes x and y have equal variance across 
        trials and observations.
    Parameters
    ----------
    x : numpy.ndarray
        n trials by m observations array
    y : numpy.ndarray
        n trials by m observations array
  
    Returns
    -------
    r2c : an estimate of the r2 between the expected values
    r2 : the fraction explained variance between the mean observations
    --------
    """
    
    n,m = np.shape(y)[-2:]
    x = np.mean(x, -2, keepdims=True)
    x_ms = x - np.mean(x, -1, keepdims=True)
    
    y = np.mean(y, -2, keepdims=True)
    y_ms = y - np.mean(y, -1, keepdims=True)
    
    xy2 = np.sum((x_ms*y_ms), -1, keepdims=True)**2
    x2 = np.sum(x_ms**2, -1, keepdims=True)
    y2 = np.sum(y_ms**2, -1, keepdims=True)
    x2y2 = x2*y2
    
    ub_xy2 = xy2 - (sig2_hat/n*(x2 + y2 - (m-1)*sig2_hat/n))
    ub_x2y2 = x2y2 - (m-1)*sig2_hat/n*(x2 + y2 - (m-1)*sig2_hat/n)
    
    r2c = ub_xy2/ub_x2y2
    
    return r2c, xy2/x2y2

visp_np_s = visp_np**0.5
u = (visp_np_s.stack(u=('unit_id', 'rec')).dropna('u', how='all'))
u = u.dropna('trial', how='any')
u = u.rename({'u':'unit', 'c':'stim'})
u = u.transpose('unit', 'trial', 'stim')
asnr_np = (u.var('trial').mean('stim')).values
asnr_np = (asnr_np[:,np.newaxis]*asnr_np[np.newaxis])
asnr_np = ravel_cor(asnr_np)**0.5

u = u[:,:-1].values.reshape((1955, 2, 2299))
a,b = ravel_cor_inds(1955)
#%%
step = 100000
rs = []

for i,j in zip(list(range(0,len(a),step)), list(range(step,len(a),step))):
    print(i/len(a))
    rs.append(np.array((r2c_n2n_sig(u[a[i:j],0][:, np.newaxis], 
                                    u[b[i:j], 1][:, np.newaxis], 
                                    sig2_hat=asnr_np[a[i:j], np.newaxis,np.newaxis]))).squeeze())
rs.append(np.array((r2c_n2n_sig(u[a[j:],0][:, np.newaxis], 
                                    u[b[j:], 1][:, np.newaxis], 
                                    sig2_hat=asnr_np[a[j:], np.newaxis,np.newaxis]))).squeeze())
rs = np.concatenate(rs,1)
#%%
u = (visp_np_s.stack(u=('unit_id', 'rec')).dropna('u', how='all'))
u = u.dropna('trial', how='any')
u = u.rename({'u':'unit', 'c':'stim'})
u = u.transpose('unit', 'trial', 'stim')
asnr_np = (u.mean('trial').var('stim')).values/(u.var('trial').mean('stim')).values
asnr_np = (asnr_np[:,np.newaxis]*asnr_np[np.newaxis])
asnr_np = ravel_cor(asnr_np)**0.5


plt.scatter(rs[:, asnr_np>2][1,::1],  rs[:, asnr_np>2][0,::1], s=1);
plt.ylim(0,1);plt.xlim(0,1);

print(np.median(rs[:, asnr_np>3.1][0,::1]))
print(np.median(rs[:, asnr_np>3.1][1,::1]))

#%%
bins = np.percentile(asnr_np, np.linspace(20,100, n_perc_bins))

da_rs = xr.DataArray(rs[:, np.argsort(asnr_np)[::1]], dims=('m', 'unit'),
             coords=[range(2), (np.sort(asnr_np)[::1])])

avr2c = da_rs[0].groupby_bins('unit', bins).median(skipna=True)
g = [a[1].values for a in list(da_rs[0].groupby_bins('unit', bins))]
avr2c_sd = np.array([bs_med_var(x, bs) for x in g])



avr2 = da_rs[1].groupby_bins('unit', bins).median(skipna=True)
g = [a[1].values for a in list(da_rs[1].groupby_bins('unit', bins))]
avr2_sd = np.array([bs_med_var(x, bs) for x in g])


plt.title('Signal Correlation')

bins_half = [(bins[1+i]+bins[i])/2 for i in range(len(bins)-1)]
plt.errorbar(bins_half, avr2c, yerr=avr2c_sd*2)
plt.errorbar(bins_half, avr2, yerr=avr2_sd*2)

plt.semilogx()
plt.xlim(1e-2, 2.5);


plt.ylim(0, .15)

#%%
v1_s = v1.sel(t=slice(0, 2), unit=[0,1])

s = v1_s.sum('t', min_count=1)
s = s**0.5 
a,b = ravel_cor_inds(len(s.coords['unit']))




#%%
ind = asnr_np>0.5
print(ind.mean()*len(ind))
rs_s = rs[:,ind]
same_s = same[ind]
m = [rs_s[0,same_s], rs_s[0,~same_s],rs_s[1,same_s], rs_s[1,~same_s]]
m = [np.mean(am) for am in m]
sd = [rs_s[0,same_s].std(), rs_s[0,~same_s].std(),rs_s[1,same_s].std(), rs_s[1,~same_s].std()]

plt.errorbar(range(4), m, 2*np.array(sd)/len(same_s)**0.5)

#%%
m = [rs_s[0,same_s], rs_s[0,~same_s],rs_s[1,same_s], rs_s[1,~same_s]]
m = [plt.hist(am, range=(0,1),histtype='step', cumulative=True, normed=True, bins=1000) for am in m]
plt.grid()
plt.legend(['same r2c', 'dif r2c', 'same r2', 'dif r2'])
#%%
r2cs_rec = np.array(r2cs_rec)
plt.errorbar(1, r2cs_rec.mean(), 2*r2cs_rec.std()/len(r2cs_rec)**0.5)
#%%

plt.ylim(-1,1.5)
thresh=0.4
plt.scatter(rs[0][::100][rs[1][::100]>thresh],  
            rs[1][::100][rs[1][::100]>thresh], s=1)
plt.xlim(0,2)

print(np.median(rs[0][(rs[1]>thresh)*same]))
print(np.median(rs[0][(rs[1]>thresh)*(~same)]))

#%%

plt.scatter(asnr_np[::100],  
            rs[1][::100], s=1)
plt.semilogx()
plt.semilogy()
plt.xlim(0.01,10)
plt.ylim(0.00001,10)


#%%
v1_s = v1.sel(t=slice(0, 2), unit=0)

s = v1_s.sum('t', min_count=1)
s = s**0.5

ind = ((~s.isnull()).sum('trial_tot')>9).prod('dir').values
s = s[(ind==1)] 
s = s.dropna('rec', how='all').dropna('trial_tot')
s = s.drop('unit')
u = s.rename({'rec':'unit', 'trial_tot':'trial'})

#u = u.values.reshape(78,4,20)

for i in [10, 20]:
    plt.figure()
    r = r2c_n2n(u[:,::2,:i].values, u[:,1::2,:i].values)
    r = np.array(r).squeeze()
    print(np.median(r,1))
    
    
#%%
v1_s = v1.sel(t=slice(0,2), unit=[0,1])
s = v1_s.sum('t', min_count=1)
s = s**0.5    
ind = ((~s.isnull()).sum('trial_tot')>9).prod('dir').prod('unit').values
s = s[(ind==1)] 
s = s.dropna('rec', how='all').dropna('trial_tot')
s = s/s.std('trial_tot')
x = np.array(r2c_n2n(s[:, 0, 1::2].values, s[:, 1, ::2].values)).squeeze()

v = bs_med_var(x[0,~np.isnan(x[0])], bs) 
m = np.median(x[0,~np.isnan(x[0])])

sig_cor_ind = (x[1]>np.median(x[1,~np.isnan(x[1])]))
sig_cor_ind = (x[1]>0)
x = x[:,sig_cor_ind]
v = bs_med_var(x[0,~np.isnan(x[0])], bs) 
m = np.median(x[0,~np.isnan(x[0])])
   
     #%%
n_times = 9
times = np.linspace(0, 2, n_times)[:-1]

print(times)

#%%
times = np.concatenate([times[:,np.newaxis], (times+times[1]-times[0])[:,np.newaxis]],1)
print(times)
#%%
times = [[0, 0.2], [0.5,2]]
rss = []
snrs= []
for time in times:
    v1_s = v1.sel(t=slice(time[0],time[1]), unit=[0,1])
    s = v1_s.sum('t', min_count=1)
    s = s**0.5    
    ind = ((~s.isnull()).sum('trial_tot')>9).prod('dir').prod('unit').values
    s = s[(ind==1)][sig_cor_ind] 
    s = s.dropna('rec', how='all').dropna('trial_tot')
    #s = s/s.std('trial_tot')
    snr = (s.mean('trial_tot').var('dir')/s.var('trial_tot').mean('dir')).prod('unit')**0.5
    snrs.append(snr)
    r1 = (np.array(r2c_n2n(s[:, 0, 1::2].values, s[:, 1, ::2].values)).squeeze())
    r2 = (np.array(r2c_n2n(s[:, 0, ::2].values, s[:, 1, 1::2].values)).squeeze())
    r = (r1+r2)/2
    rss.append(r)
rss = np.array(rss)

#%%
plt.grid()
time_mid = [(time[0]+time[1])/2 for time in times]
avr2c_sd = np.array([bs_med_var(x[~np.isnan(x)], bs) for x in rss[:,0]])
avr2c = np.array([np.median(x[~np.isnan(x)]) for x in rss[:,0]])
plt.errorbar(time_mid, avr2c, yerr= avr2c_sd*2)
for i in range(len(times)):
    plt.scatter([time_mid[i],]*len(rss[i,0][~np.isnan(rss[i, 0])]), 
                rss[i,0][~np.isnan(rss[i,0])], c='r', alpha=0.5, s=1)
plt.ylim(-.1,1)

#%%
avr2_sd = np.array([bs_med_var(x[~np.isnan(x)], bs) for x in rss[:,1]])
avr2 = np.array([np.median(x[~np.isnan(x)]) for x in rss[:,1]])
plt.errorbar(time_mid, avr2, yerr= avr2_sd)
plt.plot([0,2],[m,m])
#%%

from scipy.stats import wilcoxon, rankdata
wilcoxon(rss[0,0], rss[1,0])

dif = rss[1,0]-rss[0,0]

plt.semilogx()
plt.ylim(0,1)
#%%
for k in range(10):
    aunit = np.argsort(rankdata(snr))[::-1][k]
    plt.figure()
    plt.title(aunit)
    plt.scatter(snr,dif);
    plt.scatter(snr[aunit],dif[aunit], c='r');
    plt.ylim(0,1)
    plt.figure()
    for time, ls, time_ind in zip(times, ['--','-'], [0,1]):
        v1_s = v1.sel(t=slice(time[0],time[1]), unit=[0,1])
        s = v1_s.sum('t', min_count=1)
        s = s**0.5    
        ind = ((~s.isnull()).sum('trial_tot')>9).prod('dir').prod('unit').values
        s = s[(ind==1)][sig_cor_ind] 
        s = s.dropna('rec', how='all').dropna('trial_tot')
        #s = s/s.std('trial_tot')
        m = s[aunit].mean('trial_tot')
        v = s[aunit].std('trial_tot')
        for unit, c in zip(range(2), [[1,0,0,1], [0,1,1,1]]):
            mx = m[unit].max()
            plt.errorbar(m.coords['dir'], m[unit]/mx, 
                         yerr=(v/np.sqrt(10))/mx, c=c, ls=ls)
    
    plt.legend(['early unit 1 r =' + str(np.round(rss[0,0, aunit], 2)), 
                'early unit 2', 
                'late unit 1 r='+ str(np.round(rss[1,0, aunit], 2)), 
                'late unit 2 '])
    
    

#%%
rs=[]
for j in range(len(s)):
    sp = s[j, ...].values

    r1 = r2c_n2n(sp[0][1::2], sp[1][::2]) 
    r2 =  r2c_n2n(sp[0][::2], sp[1][1::2])
    rs.append([(r1[0]+r2[0])/2,  (r1[1]+r2[1])/2,])
rs = np.array(rs).squeeze()


plt.scatter(rs[:,0], rs[:,1]);

plt.plot([0,1]);
plt.axis('square')
plt.xlim(-2,2);
plt.ylim(0,1);
plt.grid()

#%%

s.values.reshape((37,2))

#%%
plt.scatter(rss[1,1], rss[1,0]);
plt.scatter(np.median(rss[1,1]), np.median(rss[1,0]));

#%%
ind = (snr<1.25)*(snr>0.75)
#plt.scatter(snr[ind], rss[1,0][ind]);
#plt.scatter(snr[ind], rss[1,1][ind]);
plt.scatter(rss[1,1][ind],rss[1,0][ind] );
plt.axis('square');plt.ylim(-1,2);plt.xlim(-1,2);plt.grid()
plt.plot([-1,2],[-1,2]);

#%%
v1_s = v1.sel(t=slice(0,2), unit=[0,1])
s = v1_s.sum('t', min_count=1)
s = s**0.5    

ind = ((~s.isnull()).sum('trial_tot')>9).prod('dir').prod('unit').values
s = s[(ind==1)]
s = s.dropna('rec', how='all').dropna('trial_tot')
#s = s/(s.std('dir')+0.01)
cor_dat = pd.DataFrame(np.zeros((len(s),5)), columns=['n_cor', 'spl_hi_m', 'nospl_hi_m', 'spl_lo_m', 'nospl_lo_m'])
for rec in range(len(s)):
     u1 = s[rec, 0]
     u2 = s[rec, 1]
     u1ms = u1 - u1.mean('trial_tot')
     u2ms = u2 - u2.mean('trial_tot')
     ns = np.corrcoef(u1ms.values.ravel(),u2ms.values.ravel())[0,1]
     cor_dat['n_cor'][rec]= ns
     
     cor_dat['spl_hi_m'][rec] = ((np.corrcoef(u1[5:].values.ravel(), u2[:5].values.ravel())[0,1] + 
                     np.corrcoef(u1[:5].values.ravel(), u2[5:].values.ravel())[0,1])/2)
     cor_dat['nospl_hi_m'][rec] = ((np.corrcoef(u1[:5].values.ravel(), u2[:5].values.ravel())[0,1] + 
                     np.corrcoef(u1[5:].values.ravel(), u2[5:].values.ravel())[0,1])/2)
     
     cor_dat['spl_lo_m'][rec] = ((np.corrcoef(u1[5:].mean('trial_tot').values.ravel(), 
                                     u2[:5].mean('trial_tot').values.ravel())[0,1] + 
                     np.corrcoef(u1[:5].mean('trial_tot').values.ravel(), 
                                 u2[5:].mean('trial_tot').values.ravel())[0,1])/2)
     
     cor_dat['nospl_lo_m'][rec] = ((np.corrcoef(u1[:5].mean('trial_tot').values.ravel(), 
                                 u2[:5].mean('trial_tot').values.ravel())[0,1] + 
                     np.corrcoef(u1[5:].mean('trial_tot').values.ravel(), 
                                 u2[5:].mean('trial_tot').values.ravel())[0,1])/2)
     


cor_dat = cor_dat.iloc[:78]
#%%
import r2c_common as rc
cor_dat_ci = pd.DataFrame(np.zeros(((len(s)), 4)), 
                          columns=['n_cor','r2c', 'll', 'ul'])
for rec in range(len(s)):
     print(rec/len(s))
     u1 = s[rec, 0]
     u2 = s[rec, 1]
     u1ms = u1 - u1.mean('trial_tot')
     u2ms = u2 - u2.mean('trial_tot')
     ns = np.corrcoef(u1ms.values.ravel(), u2ms.values.ravel())[0,1]
     ll, ul, r2c_hat_obs, alpha_obs = rc.r2c_ci(u1.values[:5], 
                                                u2.values[5:], alpha_targ=0.10, nr2cs=100)
     cor_dat_ci.iloc[rec, :] = np.array([ns, r2c_hat_obs, ll, ul]).squeeze()
     

#%%
ci_len = cor_dat_ci['ul'] - cor_dat_ci['ll']
thresh=0.25
ind = ci_len<thresh
l = cor_dat_ci['r2c'] - cor_dat_ci['ll']
h = cor_dat_ci['ul'] - cor_dat_ci['r2c']   
plt.errorbar(x=(cor_dat['spl_lo_m']**2)[ind], 
             y=cor_dat_ci['r2c'][ind], 
             yerr=[l[ind],h[ind]], fmt='o')
plt.legend(['90% CI'])
plt.axis('square')
plt.xlabel(r'$\hat{r}^2$')
plt.ylabel(r'$\hat{r}^2_{ER}$')
plt.title('Signal Correlation')

#plt.ylim(0,1)

#plt.xlim(0,1)
plt.plot([0,1], c='k')
plt.grid()

near_perf = ((cor_dat_ci['ul'][ind]==1))

(cor_dat_ci['ll'])[ind][near_perf]
(cor_dat_ci['ul'])[ind][near_perf]
plt.annotate('n = ' + str(np.sum(ind)) + ' neuron pairs CI<' + str(thresh), (0.2,0.05))
np.mean((cor_dat['spl_lo_m']**2)[ind]>cor_dat_ci['r2c'][ind])
plt.tight_layout()
plt.savefig('./figs/r2er_vs_r2_sig_cor.pdf')


#%%
snrs = []
snrs.append((s.mean('trial_tot').var('dir')/s.var('trial_tot').mean('dir')).prod('unit'))
snrs.append(s.mean('trial_tot').var('dir').prod('unit'))
snrs.append(s.var('trial_tot').mean('dir').prod('unit')**(-1))
snrs.append(1-ci_len)

        
#%%
s_sub = s[:78]
ci_len = cor_dat_ci['ul'] - cor_dat_ci['ll']
thresh=0.25
ind = (ci_len<thresh).values
s_sub = s_sub[ind]
r2c_sub = cor_dat_ci[ind]['r2c'].values
sort_inds = np.argsort(r2c_sub)
#%%
for i, aind in enumerate(sort_inds):
    plt.figure()
    sign = np.sign(np.corrcoef(s_sub[aind,:].mean('trial_tot'))[0,1])
    plt.title(str(np.round(sign*cor_dat_ci.iloc[ind]['r2c'].values[aind],2)) + ' ' +str(aind))
    a = (s_sub[aind,:].mean('trial_tot').T-
         s_sub[aind,:].mean('trial_tot').T.min('dir'))
    a = a/a.max('dir')
    plt.plot(s_sub[aind,:].mean('trial_tot').T, '-o')
    #plt.ylim(0,1)    


#%%
w = pd.read_csv('/home/dean/Desktop/modules/r2c/data/wyeth_r_cc_mt_vals.dat', 
            header=None, delim_whitespace=True)
w.columns = ['nms', 'r_s', 'r_n']
cor_dat['nms'] = s.coords['nms'].values[:78]
missed = []
w_l = []
the_nms = []
for j in range(len(cor_dat)):
    row = cor_dat.iloc[j]
    got_it=False
    for i in range(len(w)):
        _ = w.iloc[i]
        #print((_['nms'], nm))
        if _['nms'] in row['nms']:
            got_it = True
            the_nms.append(_['nms'])
            w_l.append([_['r_n'], _['r_s'], 
                        row['n_cor'], 
                        row['spl_lo_m'],
                        row['spl_hi_m'],
                        row['nospl_lo_m'],
                        row['nospl_hi_m'],
                        cor_dat_ci.iloc[j]['r2c'],
                        cor_dat_ci.iloc[j]['ll'], 
                        cor_dat_ci.iloc[j]['ul'],
                        snrs[0][j].values,
                        snrs[1][j].values,
                        snrs[2][j].values, 
                        snrs[3][j]])
    if got_it==False:
        missed.append(row['nms'])
w_l = np.array(w_l)
cor_da = pd.DataFrame(w_l, columns=['w_r_n', 'w_r_s', 'n_cor', 
                                    'spl_lo_m', 'spl_hi_m', 
                                    'nospl_lo_m', 'nospl_hi_m', 
                         'r2c', 'll', 'ul','snr'
                           , 'dyn', 'var', 'cilen'],
    index=the_nms)
#%%
#s.coords['rec'] = s.coords['nms']
ind = (cor_da['ul'] - cor_da['ll'])<0.25
sort_inds = cor_da[ind==1]['r2c'].sort_values().index.values


for i, aind in enumerate(sort_inds):
    print(aind)
    s.loc[aind]

#%%
from scipy import stats
plt.figure(figsize=(4,4))

n_cor_nm = 'n_cor'


plt.subplot(2,2,2)
plt.axis('square')

plt.scatter( cor_da['spl_hi_m'], cor_da[n_cor_nm],s=1)
plt.title('High m, split-trial')
r,p = stats.pearsonr(cor_da['spl_hi_m'], cor_da[n_cor_nm])
plt.title('High m, split-trial\nr=' + str(np.round(r**2,2)) + ' p=' + str(np.round(p,5)))
plt.xlim(-1,1);plt.ylim(-1,1)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.grid()

plt.subplot(2,2,1)
plt.axis('square')

plt.scatter( cor_da['nospl_hi_m'], cor_da[n_cor_nm],s=1)
plt.title('High m, no split-trial')
r, p = stats.pearsonr(cor_da['nospl_hi_m'], cor_da[n_cor_nm])
plt.title('High m, no split-trial\nr=' + str(np.round(r**2,2))+ ' p=' + str(np.round(p,5)))

plt.xlim(-1,1);plt.ylim(-1,1)

plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.grid()

plt.subplot(2,2,4)
plt.axis('square')
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.scatter(cor_da['spl_lo_m'], cor_da[n_cor_nm], s=1)
plt.xlim(-1,1);plt.ylim(-1,1)
r,p = stats.pearsonr(cor_da['spl_lo_m'], cor_da[n_cor_nm])
plt.title('Low m, split-trial\n r^2=' + str(np.round(r**2,2))+ ' p=' + str(np.round(p,5)))

plt.grid()

plt.subplot(2,2,3)
plt.axis('square')

plt.scatter(cor_da['nospl_lo_m'], cor_da[n_cor_nm],s=1)

plt.xlim(-1,1);plt.ylim(-1,1)
r, p = stats.pearsonr(cor_da['nospl_lo_m'], cor_da[n_cor_nm])
plt.title('Low m, no split-trial\nr=' + str(np.round(r**2,2))+ ' p=' + str(np.round(p,5)))

plt.grid()

plt.xlabel('Signal correlation');
plt.ylabel('Noise correlation')
plt.tight_layout()

plt.savefig('./figs/sig_cor_noise_cor.pdf')    
    
#%%
titles = ['SNR', 'Dyn Range', 'Var.', 'CI. Len']
nms = ['snr', 'dyn', 'var', 'cilen']
noise_cor_nm = 'n_cor'
#noise_cor_nm = 'w_r_n'

for i in range(4):
    ind = (cor_da[nms[i]]<np.median(cor_da[nms[i]])).values
    plt.subplot(2, 4, i+1)
    plt.scatter((cor_da['spl_lo_m'])[ind], (cor_da[noise_cor_nm])[ind], s=3)
    r, p = np.round(stats.pearsonr((cor_da['spl_lo_m'])[ind], 
                                   (cor_da[noise_cor_nm])[ind]),2)
    plt.title(titles[i]+'\nr=' + str(r))
    plt.axis('square')
    plt.ylim(-1,1)
    plt.xlim(-1,1)
    plt.grid()
    plt.gca().set_xticklabels([]);plt.gca().set_yticklabels([]);
    if i==0:
        plt.ylabel('More noisy')
for i in range(4):
    ind = ~(cor_da[nms[i]]<np.median(cor_da[nms[i]])).values
    plt.subplot(2, 4, i+5)
    plt.scatter((cor_da['spl_lo_m'])[ind], (cor_da[noise_cor_nm])[ind],s=3)
    r, p = np.round(stats.pearsonr((cor_da['spl_lo_m'])[ind], (cor_da[noise_cor_nm])[ind]),2)
    plt.title('r='+str(r))
    print((r,p))
    plt.axis('square')
    plt.ylim(-1,1)
    plt.xlim(-1,1)
    plt.grid()
    if i==0:
        plt.ylabel('Less noisy' + '\n r_noise' )
        plt.xlabel('r_sig')
    else:
        plt.gca().set_xticklabels([]);plt.gca().set_yticklabels([]); 
    
#%%
ind = (cor_da[nms[i]]<np.median(cor_da[nms[i]])).values
print(stats.spearmanr(cor_da['n_cor'][ind], (cor_da['w_r_n'])[ind]))  
plt.scatter()
ind = ~(cor_da[nms[i]]<np.median(cor_da[nms[i]])).values
print(stats.spearmanr(cor_da['n_cor'][ind], (cor_da['w_r_n'])[ind]))     
'''