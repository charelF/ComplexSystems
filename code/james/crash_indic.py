#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import math
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
warnings.simplefilter("ignore")

import sys
sys.path.append("..")
sys.path.append("../shared")

from wednesdaySPEED import simulation

import numba
print(numba.__version__)

# %%

## stolen from https://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array
def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        idx_copy = np.zeros(idx.shape[0] + 1)
        idx_copy[1:] = idx
        idx = idx_copy
        # If the start of condition is True prepend a 0
        ## idx = np.r_[0, idx]
    if condition[-1]:
        idx_copy = np.zeros(idx.shape[0] + 1)
        idx_copy[:-1] = idx
        idx_copy[-1] = condition.size
        idx = idx_copy

        # If the end of condition is True, append the length of the array
        ## idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx

def moving_average(x, w):
    '''
    utility function for calculating moving average
    of period w 
    '''
    return np.convolve(x, np.ones(w), 'valid') / w

def visualiseNICE(G, P, N, S, X, D, T, U, C):
    '''
    8 axis plot for visiualising the output of our model execution
    '''

    fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8) = plt.subplots(
        ncols=1, nrows=8, figsize=(12,12), sharex=True, gridspec_kw = 
        {'wspace':0, 'hspace':0.05, 'height_ratios':[2,2,1,1,1,1,1,1]}
    )
    im1 = ax1.imshow(G.T, cmap="bone", interpolation="None", aspect="auto")
    im4 = ax4.imshow(P.T, cmap="hot", interpolation="None", aspect="auto")
    amnwc = np.max(np.abs(N-initial_account_balance))  # absolute max net worth change
    vmin, vmax = initial_account_balance-amnwc, initial_account_balance+amnwc
    im5 = ax5.imshow(N.T, cmap="bwr", interpolation="None", aspect="auto", vmin=vmin, vmax=vmax)

    size = "15%"

    cax1 = make_axes_locatable(ax1).append_axes('right', size=size, pad=0.05)
    fig.colorbar(im1, cax=cax1, orientation='vertical')
    cax4 = make_axes_locatable(ax4).append_axes('right', size=size, pad=0.05)
    fig.colorbar(im4, cax=cax4, orientation='vertical')
    cax5 = make_axes_locatable(ax5).append_axes('right', size=size, pad=0.05)
    fig.colorbar(im5, cax=cax5, orientation='vertical')

    cax2 = make_axes_locatable(ax2).append_axes('right', size=size, pad=0.05)
    cax2.hist(S, orientation="horizontal", bins=np.linspace(np.min(S), np.max(S), len(S)//2))
    # cax2.hist(np.log10(S), orientation="horizontal", bins=np.logspace(np.log10(np.min(S)), np.log10(np.max(S)), len(S)//2))
    # cax2.set_xscale("log")
    # cax2.set_yscale("log")
    cax2.get_xaxis().set_visible(False)
    cax2.get_yaxis().set_visible(False)

    cax3 = make_axes_locatable(ax3).append_axes('right', size=size, pad=0.05)
    cax3.hist(X, orientation="horizontal", bins=np.linspace(np.min(X), np.max(X), len(X)//5))
    cax3.get_xaxis().set_visible(False)
    cax3.get_yaxis().set_visible(False)

    cax6 = make_axes_locatable(ax6).append_axes('right', size=size, pad=0.05)
    cax6.get_xaxis().set_visible(False)
    cax6.get_yaxis().set_visible(False)
    cax7 = make_axes_locatable(ax7).append_axes('right', size=size, pad=0.05)
    cax7.get_xaxis().set_visible(False)
    cax7.get_yaxis().set_visible(False)
    cax8 = make_axes_locatable(ax8).append_axes('right', size=size, pad=0.05)
    cax8.get_xaxis().set_visible(False)
    cax8.get_yaxis().set_visible(False)

    # for ax in (ax2,ax3):
    #     cax = make_axes_locatable(ax).append_axes('right', size=size, pad=0.05)
    #     # cax.axis('off')

    ##ax2.set_yscale("log")
    ax2.plot(S, label="S")
    Ws = [25]
    for W in Ws:
        ax2.plot(np.arange(W-1, len(S)), moving_average(S, W), label=f"MA{W}")
    ax2.grid(alpha=0.4)
    # ax2.legend(ncol=len(Ws)+1)

    ax3.bar(np.arange(len(X)), X)
    ax3.grid(alpha=0.4)

    # if D.shape[1] < 25:
    ax6.plot(np.mean(D[0],axis=1), color="C0", alpha=1, label="CA")
    ax6.plot(np.mean(D[1],axis=1), color="C1", alpha=1, label="momentum")
    ax6.plot(np.mean(D[2],axis=1), color="C2", alpha=1, label="invert")
    ax6.plot(np.max(D[0],axis=1), ":", color="C0", alpha=1, label="CA")
    ax6.plot(np.max(D[1],axis=1), ":", color="C1", alpha=1, label="momentum")
    ax6.plot(np.max(D[2],axis=1), ":", color="C2", alpha=1, label="invert")
    ax6.plot(np.min(D[0],axis=1), "--", color="C0", alpha=1, label="CA")
    ax6.plot(np.min(D[1],axis=1), "--", color="C1", alpha=1, label="momentum")
    ax6.plot(np.min(D[2],axis=1), "--", color="C2", alpha=1, label="invert")
    # ax6.plot(np.mean(D,axis=1), color="black", alpha=1)
    ax6.grid(alpha=0.4)
    # ax6.legend()


    ax7.set_yscale("symlog")
    ax7.plot(T, label="stack")
    ax7.plot(U, label="called shares")
    ax7.grid(alpha=0.4)
    ax7.legend()

    # if D.shape[1] < 25:
    #     ax6.plot(D, color="black", alpha=0.3)
    # ax6.plot(np.mean(D,axis=1), color="black", alpha=1)
    ax8.imshow(C.T, cmap="binary", interpolation="None", aspect="auto")
    # ax6.grid(alpha=0.4)
    
    ax8.set_xlabel("time")
    # ax2.set_ylabel("standardised log returns")
    ax2.set_ylabel("close price")
    ax1.set_ylabel("agents")
    ax3.set_ylabel("log return")
    ax4.set_ylabel("portfolio")
    ax5.set_ylabel("net worth")
    ax6.set_ylabel("influence (I)")
    ax7.set_ylabel("stack")
    ax8.set_ylabel("margin calls")

    # fig.colorbar(im, cax=ax4)

    plt.tight_layout()
    # plt.savefig("tmp.png", dpi=300)
    plt.show()

# %%

'''
How do the number of crashes vary with the value of A (activity)
'''

## sim parameters
sims = 1
threshold = -0.15
A_vals = 10

## structures
A_range = np.linspace(3, 5, A_vals)
res_A = np.zeros((sims, A_vals))

for i in range(sims):
    for j, val in enumerate(A_range):
        G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = True, pd = 0.05, pe = 0.01,
                ph = 0.0485, pa = 0.7, N0=1000, N1 = 100, A = val, a=1, h=1, 
                pi1 = 0.5, pi2 = 0.3, pi3 = 0.2)
        condition = X < threshold
        res_A[i, j] = contiguous_regions(condition).shape[0]
    
# %%

# plot results of above experiment 
fig = plt.figure(figsize=(12, 8))
plt.errorbar(A_range, np.mean(res_A, axis=0), yerr=np.std(res_A, axis=0), color="C4")
plt.grid(alpha=0.2)
plt.ylabel("Crashes")
plt.xlabel("A")
plt.show()

# %%

'''
How do the number of crashes vary with the value of ph?
'''

## sim parameters
sims = 10
threshold = -0.15
ph_vals = 10

## structures
ph_range = np.linspace(0.01, 0.1, ph_vals)
res_ph = np.zeros((sims, ph_vals))

for i in range(sims):
    print(i)
    for j, val in enumerate(ph_range):
        G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = True, pd = 0.05, pe = 0.01,
                ph = val, pa = 0.7, N0=1000, N1 = 100, A = 4, a=1, h=1, 
                pi1 = 0.5, pi2 = 0.3, pi3 = 0.2)
        condition = X < threshold
        res_ph[i, j] = contiguous_regions(condition).shape[0]

# %%

# plot results of above experiment 

fig = plt.figure(figsize=(12, 8))
plt.errorbar(ph_range, np.mean(res_ph, axis=0), yerr=np.std(res_ph, axis=0), color="C4")
plt.grid(alpha=0.2)
plt.ylabel("Crashes")
plt.xlabel(r"$p_h$")
plt.show()
    
# %%

'''
How do the number of crashes vary with the value of pd?
'''

## sim parameters
sims = 10
threshold = -0.15
pd_vals = 10

## structures
pd_range = np.linspace(0.03, 0.07, pd_vals)
res_pd = np.zeros((sims, pd_vals))

for i in range(sims):
    for j, val in enumerate(pd_range):
        G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = True, pd = val, pe = 0.01,
                ph = 0.0485, pa = 0.7, N0=1000, N1 = 100, A = 4, a=1, h=1, 
                pi1 = 0.5, pi2 = 0.3, pi3 = 0.2)
        condition = X < threshold
        res_pd[i, j] = contiguous_regions(condition).shape[0]

# %%

# plot results of above 
fig = plt.figure(figsize=(12, 8))
plt.errorbar(pd_range, np.mean(res_pd, axis=0), yerr=np.std(res_pd, axis=0), color="C4")
plt.grid(alpha=0.2)
plt.ylabel("Crashes")
plt.xlabel(r"$p_d$")
plt.show()

# %% 

'''
How do the number of crashes vary with the number of agents in the model?
'''

## sim parameters
sims = 15
threshold = -0.15
trader_vals = 10

## structures
trader_range = np.rint(np.linspace(10, 300, trader_vals)).astype(np.int32)
res_traders = np.zeros((sims, trader_vals))

for i in range(sims):
    for j, val in enumerate(trader_range):
        print(val)
        G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = True, pd = 0.05, pe = 0.01,
                ph = 0.0485, pa = 0.7, N0=1000, N1 = val, A = 4, a=1, h=1, 
                pi1 = 0.5, pi2 = 0.3, pi3 = 0.2)
        condition = X < threshold
        res_traders[i, j] = contiguous_regions(condition).shape[0]

# %%

# plot the results of the above 
fig = plt.figure(figsize=(12, 8))
plt.errorbar(trader_range, np.mean(res_traders, axis=0), yerr=1.96*np.std(res_traders, axis=0) / np.sqrt(sims), color="C4")
plt.grid(alpha=0.2)
plt.ylabel("Crashes")
plt.xlabel(r"Number of Agents")
plt.show()


# %%

'''
How do the number of crashes vary with the lenght of time period investigated?
'''

## sim parameters
sims = 55
threshold = -0.15
pi2_vals = 20

## structures
n_range = [10, 25, 2000]
pi2_range = np.linspace(0.01, 0.45, pi2_vals)
res_pi2 = np.zeros((sims, pi2_vals, len(n_range)))

for k, n_val in enumerate(n_range):
    for i in range(sims):
        print(i)
        for j, val in enumerate(pi2_range):
            G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = True, pd = 0.05, pe = 0.01,
                    ph = 0.0485, pa = 0.7, N0=n_val, N1 = 100, A = 4, a=1, h=1, 
                    pi1 = 0.5, pi2 = val, pi3 = 0.5 - val)
            condition = X < threshold
            res_pi2[i, j, k] = contiguous_regions(condition).shape[0]

# %%

'''
How do the number of crashes vary as we vary the proportion of stochastic traders
'''

colors = ["C3", "C7", "C9"]

fig = plt.figure(figsize=(12, 8))
for k, n_val in enumerate(n_range):

    m = np.mean(res_pi2[:,:,k], axis=0)
    ci = 1.96*np.std(res_pi2[:,:,k]) / np.sqrt(sims)
    plt.fill_between(pi2_range, m-ci, m+ci, alpha=0.2)
    plt.errorbar(pi2_range, np.mean(res_pi2[:,:,k], axis=0), yerr=1.96*np.std(res_pi2[:,:,k], axis=0) / np.sqrt(sims), color=colors[k], label=f"N={n_range[k]}")
plt.grid(alpha=0.2)
plt.legend(fontsize=14)
plt.ylabel("Crashes", fontsize=14)
plt.xlabel("pi_2", fontsize=14)
plt.savefig("imgs/pi_crashes_analog", dpi=350)
plt.show()

# %%

'''
How do the number of crashes vary as we vary the thresold value?
'''

## sim parameters
sims = 15
threshold_vals = 10

## structures
threshold_range = np.linspace(-0.05, -0.5, threshold_vals)
res_threshold = np.zeros((sims, threshold_vals))

for i in range(sims):
    G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = True, pd = 0.05, pe = 0.01,
                ph = 0.0485, pa = 0.7, N0=1000, N1 = 100, A = 4, a=1, h=1, 
                pi1 = 0.5, pi2 = 0.3, pi3 = 0.2)
    for j, val in enumerate(threshold_range):
        condition = X < val
        res_threshold[i, j] = contiguous_regions(condition).shape[0]

# %%

# plot results of the above experiment, with the error bars 

fig = plt.figure(figsize=(12, 8))
plt.errorbar(threshold_range, np.mean(res_threshold, axis=0), yerr=np.std(res_threshold, axis=0), color="C4")
plt.grid(alpha=0.2)
plt.ylabel("Crashes")
plt.xlabel("Threshold")
plt.show()

# %%

'''
How do the number of crashes vary with the number of 
time period in the crash defintiion?
'''


## sim parameters
sims = 10
threshold = -0.01

## structures
threshold_range = [1, 2, 3, 4, 5, 6, 7]
res_threshold = np.zeros((sims, len(threshold_range)))

for i in range(sims):
    G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = True, pd = 0.05, pe = 0.01,
                ph = 0.0485, pa = 0.7, N0=1000, N1 = 100, A = 4, a=1, h=1, 
                pi1 = 0.5, pi2 = 0.3, pi3 = 0.2)
    for j, val in enumerate(threshold_range):
        condition = X < threshold
        continuous_reg = contiguous_regions(condition)
        continuous_reg_filtered = continuous_reg[(np.max(continuous_reg, axis=1) - np.min(continuous_reg, axis=1)) > val]
        # print(i, j)
        # print(res_threshold.shape)
        # print(continuous_reg_filtered.shape)
        res_threshold[i, j] = continuous_reg_filtered.shape[0]

fig = plt.figure(figsize=(12, 8))
plt.errorbar(threshold_range, np.mean(res_threshold, axis=0), yerr=1.96*np.std(res_threshold, axis=0) / np.sqrt(sims), color="C4")
plt.grid(alpha=0.2)
plt.ylabel("Crashes")
plt.xlabel("Consecutive Periods Required for Crash")
plt.show()

# %%

'''
How do the number of crashes vary as we increase the proportion of 
moving average traders?
'''

## sim parameters
sims = 20
threshold = -0.15
pi3_vals = 10

## structures
pi3_range = np.linspace(0.05, 0.45, pi3_vals)
res_pi3 = np.zeros((sims, pi3_vals))

for i in range(sims):
    for j, val in enumerate(pi3_range):
        print(val)
        G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = True, pd = 0.05, pe = 0.01,
                ph = 0.0485, pa = 0.7, N0=1000, N1 = 100, A = 4, a=1, h=1, 
                pi1 = 0.5, pi2 = 0.5 - val, pi3 = val)
        condition = X < threshold
        res_pi3[i, j] = contiguous_regions(condition).shape[0]

# %%

# plotting figure of the of the above experiment

fig = plt.figure(figsize=(12, 8))
plt.errorbar(pi3_range, np.mean(res_pi3, axis=0), yerr=1.96*np.std(res_pi3, axis=0) / np.sqrt(sims), color="C4")
plt.grid(alpha=0.2)
plt.ylabel("Crashes")
plt.xlabel(r"$pi3$")
plt.show()

#%%

## hypothesis test, does adding portfolio management to the model result in more crashes 

## sim parameters
sims = 50
prob_range = [0, 0.25]
threshold = 0.15

## structures
res_pm = np.zeros((len(prob_range), sims))

for i in range(sims):
    print(i)
    for j, val in enumerate(prob_range):
        G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = True, pd = 0.05, pe = 0.01,
                ph = 0.0485, pa = 0.7, N0=1000, N1 = 100, A = 4, a=1, h=1, 
                pi1 = 0.5, pi2 = val, pi3 = 0.5 - val)
        condition = X < threshold
        res_pm[j, i] = contiguous_regions(condition).shape[0]

#%%

print(res_pm.shape)
print(res_pm[0,:].mean())
print(res_pm[1,:].mean())
st.ttest_ind(res_pm[0,:], res_pm[1,:], alternative="greater")
