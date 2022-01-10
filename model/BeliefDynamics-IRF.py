# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Belief Dynamics on the Social Network 
#
# - This notebook explores various more generic implications of the social-network-based learning in _Learning from Friends in a Pandemic_ by Christos Makridis and Tao Wang. These results are not included in the paper.

import numpy as np
import pandas as pd
from interpolation import interp, mlinterp
from scipy import interpolate
from numba import njit, float64
from numba.experimental import jitclass
import matplotlib as mp
import matplotlib.pyplot as plt
# %matplotlib inline


# + {"code_folding": []}
## figures configurations

mp.rc('xtick', labelsize=14) 
mp.rc('ytick', labelsize=14) 


fontsize = 14
legendsize = 12
linewith= 3.0

# + {"code_folding": []}
## load SCI weight matrix 

W = np.load('SCIWeight.npy')
W16 =  np.load('SCIWeight16.npy')

N = W.shape[0]
N16 = W16.shape[0]


# -

# ## IRFs with social network learning 
#
# - plots the impulse response of average beliefs about the aggregate state following 
#   - aggregate/fundamental-relevant shock
#   - idiosyncratic/news shocks 
#     - in different locations
#     - under different networks

# + {"code_folding": [4, 29, 40]}
##############################################################
## analytical functison of IRFs to an aggregate shock and idiosycratic shocks
#########################################################################

def irf_ag(k,
           λ,
           W, ## listening matrix 
           v): ## steps of IRF
    """
    this is to operate on matrix, very slow. 
    actually it is not a function of W. So can use irf_ag_simple functions.
    but i still include it for comparison purpose 
    """
    
    N = len(W)
    H = np.ones(N)
    Iκ = (1-k)*H   ## I-\kappa
    X0 = 1/N*k*(1-λ)
    M = (1-λ)*np.diag(Iκ)+λ*W
    irf = 0.0
    for s in range(v+1):
        Ms = np.linalg.matrix_power(M,s)
        #########
        HMsH = Ms.sum()
        #############
        Xs = X0*HMsH
        irf = irf+Xs
    return irf

def irf_ag_simple(k,
                  λ,
                  v):
    x0 = k*(1-λ)
    rho =(1-λ)*(1-k)+λ  ## this is the summary statistic of the average belief 
    irf = 0.0
    for s in range(v+1):
        xv = x0*(rho)**s
        irf = irf+xv
    return irf

def irf_id_c(k,
             λ, 
             W,  ## listening matrix 
             v,  ## steps of IRF
             Z): ## new addition, a vector of 1/0s deciding wheather the shock hit there
    
    N = len(W)
    H = np.ones(N)
    
    Iκ = (1-k)*H
    X0 =1/N*k*(1-λ)
    M = (1-λ)*np.diag(Iκ)+λ*W
    Mv = np.linalg.matrix_power(M,v)
    #####
    ## different for ag and id
    HMvZ = H@Mv@Z
    ######
    Xv = X0*HMvZ
    irf = Xv
    return irf


# -

# #### How does aggregate IRF depends on k and $\lambda$

# + {"code_folding": [0]}
## compute IRFs 

"""
grid_size = 20

k_vals = np.linspace(0.001, 2, grid_size)
λ_vals = np.linspace(0.001, 0.99, grid_size)

## generate IRFs for different v
vs = [0,1,2,3]

IRF_ags_ls= []

for v_id,v in enumerate(vs):
    IRF_ags = np.empty((grid_size, grid_size))
    for i, k in enumerate(k_vals):
        for j, λ in enumerate(λ_vals):
            #IRF_ags[i, j] = irf_ag(k,λ,W,v)
            IRF_ags[i, j] = irf_ag_simple(k,
                                          λ,
                                          v)
    IRF_ags_ls.append(IRF_ags)
    
"""

# + {"code_folding": [0, 7]}
#levels = np.array([0.0,0.2,0.4,0.6,0.8,1.0,1.5])

"""
## plot 
fig, ax = plt.subplots(2,2,figsize=(13,10))
ax = ax.ravel()

for v_id,v in enumerate(vs):
    this_IRF_ags = IRF_ags_ls[v_id]
    cs1 = ax[v_id].contourf(k_vals, λ_vals, this_IRF_ags.T, alpha=0.2)
    ctr1 = ax[v_id].contour(k_vals, λ_vals, this_IRF_ags.T)
    plt.clabel(ctr1, inline=1, fontsize=13)
    #plt.colorbar(cs1, ax=ax)
    ax[v_id].set_title(r'IRF, v={}'.format(v))
    ax[2].set_xlabel("$k$", fontsize=16)
    ax[3].set_xlabel("$k$", fontsize=16)
    ax[0].set_ylabel("$λ$", fontsize=16)
    ax[2].set_ylabel("$λ$", fontsize=16)
    
    ax[v_id].ticklabel_format(useOffset=False)

    ax[v_id].annotate('overreactive', xy=(1.8,0.1),
                xytext=(1.0,0.1), va='center', multialignment='right',
                arrowprops={'arrowstyle': '->', 'lw': 2, 'ec': 'r'})
    
    ax[v_id].annotate('rigid', xy=(0.1,0.1),
                xytext=(0.4,0.1), va='center', multialignment='right',
                arrowprops={'arrowstyle': '->', 'lw': 2, 'ec': 'r'})

    ax[v_id].annotate('social', xy=(0.2,0.7),
                xytext=(0.1,0.5), va='center', multialignment='right',
                arrowprops={'arrowstyle': '->', 'lw': 2, 'ec': 'r'})
    

plt.savefig('../graph/model/IRF_ag_contourf.jpg')

"""
# -

# #### The time path of the IRF

k_fix = 0.33
k_vals = np.array([0.3,0.5,0.9])
λ_vals = np.array([0.01,0.3,0.8,0.99])
v_vals = np.arange(10)

# + {"code_folding": [7]}
IRF_ag_path = np.empty((len(k_vals),len(λ_vals), len(v_vals)))
                       
for j,k in enumerate(k_vals):
    for i, λ in enumerate(λ_vals):
        for t, v in enumerate(v_vals):
            IRF_ag_path[j,i,t] = irf_ag_simple(k,
                                              λ,
                                              v)

# + {"code_folding": [8]}
lp_ls=['-','--','-.','v-']
nb_plots = len(k_vals)

fig,ax =plt.subplots(nb_plots,1,figsize=(7,10))
ax = ax.ravel()

for j,k in enumerate(k_vals):
    for i,λ in enumerate(λ_vals):
        ax[j].plot(IRF_ag_path[j,i,:],
                lp_ls[i],
               label =r'λ={}'.format(λ))
        ax[j].set_xlabel('v')
        ax[j].set_ylabel(r'$IRF^{ag}$')
        ax[j].set_title(r'k={}'.format(k))
        ax[j].set_xticks(v_vals)
        ax[j].grid('major',color='gray', linestyle='--', linewidth=1)
        ax[j].legend(loc=0)
fig.tight_layout(pad=1.4)
plt.savefig('../graph/model/IRF_ag_path.jpg')


# -

# #### How does idiosyncratic IRF depends on k and $\lambda$
#
# - we assume the shocks happen at the top x fraction of influenial nodes 

# + {"code_folding": [0]}
def where_to_shock(W,
                  where):  # a tuple, for instance (0.0,0.33) represents the top third influencers to be hit
    N = len(W)
    Z = np.zeros(N)
    degree = np.sum(W,axis = 0)
    rank_idx = np.flip(degree.argsort())  ## descending sorted index, e.g. the first element is the index of smallest influence
    lb,ub = where
    cut_lb,cut_ub = np.int(N*lb),np.int(N*ub),
    shocked_idx = rank_idx[cut_lb:cut_ub-1]
    Z[shocked_idx] = 1
    return Z 


# + {"code_folding": []}
## generate Z vectors 
Z_top = where_to_shock(W,
                      (0.0,0.33))
Z_mid = where_to_shock(W,
                      (0.33,0.66))
Z_bot = where_to_shock(W,
                      (0.66,0.99))

# + {"code_folding": []}
"""
## compute IRFs 

grid_size = 20

k_vals = np.linspace(0.001, 2, grid_size)
λ_vals = np.linspace(0.001, 0.99, grid_size)


## generate IRFs for different v
vs = [0,1,2,3]
IRF_ids_ls = []

for v_id,v in enumerate(vs):
    IRF_ids = np.empty((grid_size, grid_size))
    for i, k in enumerate(k_vals):
        for j, λ in enumerate(λ_vals):
            IRF_ids[i, j] = irf_id_c(k=k,
                                     λ=λ,
                                     W=W,
                                     v=v,
                                    Z = Z_top)
    IRF_ids_ls.append(IRF_ids)
    
"""

# + {"code_folding": [7]}
"""
## plot 
#levels = np.array([0.0,0.1,0.2,0.3,0.4,0.5])

fig, ax = plt.subplots(2,2,figsize=(13,10))
ax = ax.ravel()


for v_id,v in enumerate(vs):
    this_IRF_ids = IRF_ids_ls[v_id]
    cs1 = ax[v_id].contourf(k_vals, λ_vals, this_IRF_ids.T, alpha=0.2)
    ctr1 = ax[v_id].contour(k_vals, λ_vals, this_IRF_ids.T)
    plt.clabel(ctr1, inline=2, fontsize=13)
    #plt.colorbar(cs1, ax=ax)
    ax[2].set_xlabel("$k$", fontsize=16)
    ax[3].set_xlabel("$k$", fontsize=16)
    ax[0].set_ylabel("$λ$", fontsize=16)
    ax[2].set_ylabel("$λ$", fontsize=16)
    ax[v_id].set_title(r'IRF, v={}'.format(v))

    ax[v_id].ticklabel_format(useOffset=False)

    ax[v_id].annotate('overreactive', xy=(1.8,0.1),
                xytext=(1.0,0.1), va='center', multialignment='right',
                arrowprops={'arrowstyle': '->', 'lw': 2, 'ec': 'r'})
    
    ax[v_id].annotate('rigid', xy=(0.1,0.1),
                xytext=(0.4,0.1), va='center', multialignment='right',
                arrowprops={'arrowstyle': '->', 'lw': 2, 'ec': 'r'})

    ax[v_id].annotate('social', xy=(1.68,0.5),
                xytext=(1.6,0.3), va='center', multialignment='right',
                arrowprops={'arrowstyle': '->', 'lw': 2, 'ec': 'r'})

plt.savefig('../graph/model/IRF_id_contourf.jpg')


"""
# -

# #### Time path of IRF to local shocks to the top

k_fix = 0.33
k_vals = np.array([0.3,0.5,0.9])
λ_vals = np.array([0.01,0.3,0.8,0.99])
v_vals = np.arange(10)

# + {"code_folding": [2]}
IRF_id_path = np.empty((len(k_vals),len(λ_vals), len(v_vals)))
                       
for j,k in enumerate(k_vals):
    for i, λ in enumerate(λ_vals):
        for t, v in enumerate(v_vals):
            IRF_id_path[j,i,t] = irf_id_c(k=k,
                                          λ=λ,
                                          W=W,
                                          v=v,
                                          Z = Z_top)

# + {"code_folding": [8]}
lp_ls=['-','--','-.','v-']
nb_plots = len(k_vals)
#b_lim =-0.1 ## y axis range lower lim
#t_lim = 0.6  ## y axis range upper lim

fig,ax =plt.subplots(nb_plots,1,figsize=(7,10))
ax = ax.ravel()

for j,k in enumerate(k_vals):
    for i,λ in enumerate(λ_vals):
        ax[j].plot(IRF_id_path[j,i,:],
                lp_ls[i],
               label =r'λ={}'.format(λ))
        ax[j].set_xlabel('v')
        ax[j].set_ylabel(r'$IRF^{id}$')
        ax[j].set_title(r'k={}, top'.format(k))
        ax[j].set_xticks(v_vals)
        ax[j].grid('major',color='gray', linestyle='--', linewidth=1)
        ax[j].legend(loc=0)
        #ax[j].set_ylim(b_lim, t_lim)
fig.tight_layout(pad=1.4)
plt.savefig('../graph/model/IRF_id_path_top.jpg')
# -

# #### Time path of IRF to local shocks to the bottom

# + {"code_folding": [7]}
k_fix = 0.33
k_vals = np.array([0.3,0.5,0.9])
λ_vals = np.array([0.01,0.3,0.8,0.99])
v_vals = np.arange(10)

IRF_id_path_bot = np.empty((len(k_vals),len(λ_vals), len(v_vals)))
                       
for j,k in enumerate(k_vals):
    for i, λ in enumerate(λ_vals):
        for t, v in enumerate(v_vals):
            IRF_id_path_bot[j,i,t] = irf_id_c(k=k,
                                          λ=λ,
                                          W=W,
                                          v=v,
                                          Z = Z_bot)

# + {"code_folding": [9]}
lp_ls=['-','--','-.','v-']
nb_plots = len(k_vals)

#b_lim =-0.3 ## y axis range lower lim
#t_lim = 0.6  ## y axis range upper lim

fig,ax =plt.subplots(nb_plots,1,figsize=(7,10))
ax = ax.ravel()

for j,k in enumerate(k_vals):
    for i,λ in enumerate(λ_vals):
        ax[j].plot(IRF_id_path_bot[j,i,:],
                lp_ls[i],
               label =r'λ={}'.format(λ))
        ax[j].set_xlabel('v')
        ax[j].set_ylabel(r'$IRF^{id}$')
        ax[j].set_title(r'k={}, bottom'.format(k))
        ax[j].set_xticks(v_vals)
        ax[j].grid('major',color='gray', linestyle='--', linewidth=1)
        ax[j].legend(loc=0)
        #ax[j].set_ylim(b_lim, t_lim)
fig.tight_layout(pad=1.4)
plt.savefig('../graph/model/IRF_id_path_bot.jpg')
# -

# ### How does IRF depend on network structure 

autarky = np.identity(N)
egal = np.ones((N,N))*1/N

# + {"code_folding": [1]}
## different networks 
W_ls = [W,
        autarky,
        egal]
labels_ls =['Actual','Autarky','Egalitarian']

IRF_ids_ls=[]

## compute IRFs 

grid_size = 20

k_vals = np.linspace(0.001, 2.00, grid_size)
λ_vals = np.linspace(0.001, 0.99, grid_size)


for s,weigt in enumerate(W_ls):
    IRF_ids = np.empty((grid_size, grid_size))
    for i, k in enumerate(k_vals):
        for j, λ in enumerate(λ_vals):
            IRF_ids[i, j] = irf_id_c(k=k,
                                     λ=λ,
                                     W=weigt,
                                     v=1,
                                     Z = Z_top)
    # foreach weight matrix 
    IRF_ids_ls.append(IRF_ids)


# + {"code_folding": []}
## plot 
fig, ax = plt.subplots(1,3,figsize=(18,5))

ax = ax.ravel()

for plt_id in range(len(W_ls)):
    IRF_ids_this = IRF_ids_ls[plt_id]
    cs1 = ax[plt_id].contourf(k_vals, λ_vals, IRF_ids_this.T, alpha=0.2)
    ctr1 = ax[plt_id].contour(k_vals, λ_vals, IRF_ids_this.T)


    ax[plt_id].set_title(str(labels_ls[plt_id]))
    ax[plt_id].set_xlabel("$k$", fontsize=16)
    ax[0].set_ylabel("$λ$", fontsize=16)

    ax[plt_id].ticklabel_format(useOffset=False)


    ax[plt_id].annotate('overreactive', xy=(1.8,0.1),
                xytext=(1.0,0.1), va='center', multialignment='right',
                arrowprops={'arrowstyle': '->', 'lw': 2, 'ec': 'r'})
    
    ax[plt_id].annotate('rigid', xy=(0.1,0.1),
                xytext=(0.4,0.1), va='center', multialignment='right',
                arrowprops={'arrowstyle': '->', 'lw': 2, 'ec': 'r'})

    ax[plt_id].annotate('social', xy=(1.68,0.5),
                xytext=(1.6,0.3), va='center', multialignment='right',
                arrowprops={'arrowstyle': '->', 'lw': 2, 'ec': 'r'})

plt.clabel(ctr1, inline=1, fontsize=13)
plt.colorbar(cs1, ax=ax)
plt.savefig('../graph/model/IRF_id_W_compare.jpg')

# + {"code_folding": []}
## time path 
Z_top_16 = where_to_shock(W16,
                          (0.0,0.33))
# -

k_fix = 0.33
k_vals = np.array([0.3,0.5,0.9])
λ_vals = np.array([0.01,0.3,0.8,0.99])
v_vals = np.arange(10)

# + {"code_folding": [2]}
IRF_id_path_16 = np.empty((len(k_vals),len(λ_vals), len(v_vals)))
                       
for j,k in enumerate(k_vals):
    for i, λ in enumerate(λ_vals):
        for t, v in enumerate(v_vals):
            IRF_id_path_16[j,i,t] = irf_id_c(k=k,
                                             λ=λ,
                                             W=W16,
                                             v=v,
                                             Z = Z_top_16)
            


# + {"code_folding": [8]}
lp_ls=['-','--','-.','v-']
nb_plots = len(k_vals)
#b_lim =0.0 ## y axis range lower lim
#t_lim = 0.6  ## y axis range upper lim

fig,ax =plt.subplots(nb_plots,1,figsize=(7,10))
ax = ax.ravel()

for j,k in enumerate(k_vals):
    for i,λ in enumerate(λ_vals):
        ax[j].plot(IRF_id_path_16[j,i,:],
                lp_ls[i],
               label =r'λ={}'.format(λ))
        ax[j].set_xlabel('v')
        ax[j].set_ylabel(r'$IRF^{id}$')
        ax[j].set_title(r'k={}, top, 2016'.format(k))
        ax[j].set_xticks(v_vals)
        ax[j].grid('major',color='gray', linestyle='--', linewidth=1)
        ax[j].legend(loc=0)
        #ax[j].set_ylim(b_lim, t_lim)
fig.tight_layout(pad=1.4)
plt.savefig('../graph/model/IRF_id_path_top_2016.jpg')
