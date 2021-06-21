#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit, prange

import sys
sys.path.append('../shared')

from wednesdaySPEED import *
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#----------------------CRASH ANALYSIS-----------------------------|
@jit(nopython=True)
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
        idx_copy = np.zeros((idx.shape[0] + 1), dtype = np.int64)
        idx_copy[1:] = idx
        idx = idx_copy
        # If the start of condition is True prepend a 0
        ## idx = np.r_[0, idx]
    if condition[-1]:
        idx_copy = np.zeros((idx.shape[0] + 1), dtype = np.int64)
        idx_copy[:-1] = idx
        idx_copy[-1] = condition.size
        idx = idx_copy

    # Reshape the result into two columns
    idx= np.reshape(idx,(-1,2))
    return idx

@jit(nopython=True)
def count_crashes(X, treshold, window=5):
    """
    does it better than james
    - X: log returns array, in range -1, 1
    - treshold: the log return that defines a crash: 
        - e.g. if 20% drop over 5 days = crash then the treshold should be 0.8
    - window: how many days: default: 5 days
    """

    crashes = 0
    for i in range(len(X)-window):
        period = Xp1[i:i+window]+1
        prod = np.prod(period)
        geo_mean = prod ** (1/window)
        if geo_mean < treshold:
            crashes += 1

    return crashes

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def visualiseNICE(G, P, N, S, X, D, T, U, C):
    fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8) = plt.subplots(
        ncols=1, nrows=8, figsize=(12,12), sharex=True, gridspec_kw = 
        {'wspace':0, 'hspace':0.05, 'height_ratios':[1,2,1,1,1,1,1,1]}
    )
    im1 = ax1.imshow(G.T, cmap="bwr", interpolation="None", aspect="auto")
    im4 = ax4.imshow(P.T, cmap="bwr", interpolation="None", aspect="auto")
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

    # ax2.set_yscale("log")
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = False, pd = 0.05, pe = 0.01,
        ph = 0.0485, pa = 0.7, N0=1000, N1 = 1000, A =0, a=1, h=1, 
        pi1 = 0.5, pi2 = 0.3, pi3 = 0.2)

visualiseNICE(G,P,N,S,X,D,T,U,C)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------PHASE TRANSITION-------------------#
series = np.load("../../data/ENTROPYPLOT/E1_S_timeseries.npy")
tau = 9
N = 100
# series = S
splt = np.array_split(series, N)
q_vals = np.linspace(-4, 4, 100)

## structs
C_q = np.zeros(q_vals.shape[0] - 2) 
X_q = np.zeros(q_vals.shape[0])
S_q = np.zeros(q_vals.shape[0] - 1)
mu_i = np.zeros(len(splt))
denom_sum = 0

## eq 10
for i in range(len(splt)):
    denom_sum += np.abs(splt[i][tau] - splt[i][0])

for j in range(len(splt)):
    mu_i[j] = np.abs(splt[j][tau] - splt[j][0]) / denom_sum

lhs = np.zeros((q_vals.shape[0]))
rhs = np.zeros((q_vals.shape[0]))

for k, val in enumerate(q_vals):
    ## eq 11
    lhs[k] = np.log(np.sum(mu_i**val))
    rhs[k] = np.log(N)
    ## solve for slope of log-log
    ## x_q equivelent to tau(q) in casenna
    X_q[k] = lhs[k] / rhs[k]

# ## cannot obtain C_q for first and last q vals
for l in range(1, q_vals.shape[0] - 1):
    C_q[l - 1] = X_q[l + 1] - 2 * X_q[l] + X_q[l - 1]
    S_q[l - 1] = X_q[l + 1] - X_q[l - 1]


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plt.figure(figsize=(10,5))
plt.plot(q_vals/40, X_q/np.max(X_q), c="r", label="Free Energy - H")
plt.plot(q_vals[2:]/40, S_q[:-1]/np.max(-S_q), c="b", label="Entropy - dH/dT")
plt.plot(q_vals[2:]/40,C_q/np.max(C_q), c="g", label="Specific heat- dH^2/dT^2")
plt.ylabel("")
plt.xlabel("Temperature")
plt.legend()
plt.show()


plt.figure(figsize=(10,5))
plt.plot(q_vals, X_q)
plt.ylabel("H - Free Energy")
plt.xlabel("Temperature")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(q_vals[2:], S_q[:-1])
plt.ylabel("S - Entropy")
plt.xlabel("Temperature")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(q_vals[2:],C_q)
plt.ylabel("C_p - Specific heat")
plt.xlabel("Temperature")
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@njit(parallel=True)
def parallel_simulation_phase_transition(PAR1_range,PAR2_range, PAR3_range,SIM, threshold, N0):
    
    crashes = np.zeros((len(PAR1_range), SIM), dtype=np.float64)
    S_arrays = np.zeros((len(PAR1_range), SIM, N0), dtype=np.float64)

    for i in prange(len(PAR1_range)):
        PAR1_VAL = PAR1_range[i]
        PAR2_VAL = PAR2_range[i]
        PAR3_VAL = PAR3_range[i]
        for j in prange(SIM):
            G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = True, pd = 0.05, pe = 0.01,
                    ph = 0.0485, pa = 0.7, N0=N0, N1 = 100, A =4, a=1, h=1, 
                    pi1 = PAR1_VAL, pi2 = PAR2_VAL, pi3 = PAR3_VAL)
            # CRASH DATA         
            condition = X < threshold
            crashes[i,j] = contiguous_regions(condition).shape[0]
            S_arrays[i,j] = S

    return (crashes, S_arrays)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

PAR1_range = np.linspace(0, 1 ,50)
PAR2_range = (1-PAR1_range)*0.3
PAR3_range = (1-PAR1_range)*0.2

SIM = 100
threshold = 0.2
N0 = 1000

crashes, S_ARRAY = parallel_simulation_phase_transition(PAR1_range,PAR2_range,PAR3_range, SIM, threshold, N0)

crashes_mean = np.mean(crashes, axis=1)
crashes_std = np.std(crashes, axis=1)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tau = 9
N = 100

q_vals = np.linspace(-5, 5, 1000)
C_q = np.zeros(q_vals.shape[0] - 2) 
X_q = np.zeros(q_vals.shape[0])
S_q = np.zeros(q_vals.shape[0] - 1)
mu_i = np.zeros(len(splt))
lhs = np.zeros((q_vals.shape[0]))
rhs = np.zeros((q_vals.shape[0]))



C_q_collector = np.empty((len(PAR1_range), SIM, *C_q.shape))
X_q_collector = np.empty((len(PAR1_range), SIM, *X_q.shape))
S_q_collector = np.empty((len(PAR1_range), SIM, *S_q.shape))


for i_par,par in enumerate(PAR1_range):
    print(i_par)
    for sim in range(SIM):
        series = S_ARRAY[i_par, sim]
        splt = np.array_split(series, N)
        
        ## structs
        denom_sum = 0

        ## eq 10
        for i in range(len(splt)):
            denom_sum += np.abs(splt[i][tau] - splt[i][0])

        for j in range(len(splt)):
            mu_i[j] = np.abs(splt[j][tau] - splt[j][0]) / denom_sum


        for k, val in enumerate(q_vals):
            ## eq 11
            lhs[k] = np.log(np.sum(mu_i**val))
            rhs[k] = np.log(N)
            ## solve for slope of log-log
            ## x_q equivelent to tau(q) in casenna
            X_q[k] = lhs[k] / rhs[k]

        # ## cannot obtain C_q for first and last q vals
        for l in range(1, q_vals.shape[0] - 1):
            C_q[l - 1] = X_q[l + 1] - 2 * X_q[l] + X_q[l - 1]
            S_q[l - 1] = X_q[l + 1] - X_q[l - 1]

        C_q_collector[i_par, sim] = C_q
        X_q_collector[i_par, sim] = X_q
        S_q_collector[i_par, sim] = S_q

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C_q_collector_no_nan = np.nan_to_num(C_q_collector, nan=0)
C_q_mean = np.mean(C_q_collector_no_nan, axis=1)
print(C_q_mean.shape)

X_q_collector_no_nan = np.nan_to_num(X_q_collector, nan=0)
X_q_mean = np.mean(X_q_collector_no_nan, axis=1)
print(X_q_mean.shape)

S_q_collector_no_nan = np.nan_to_num(S_q_collector, nan=0)
S_q_mean = np.mean(S_q_collector_no_nan, axis=1)
print(S_q_mean.shape)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plt.figure(figsize=(6, 4))
plt.imshow(C_q_mean, aspect="auto", interpolation="None", vmin=0, vmax=0.01)
plt.colorbar()
plt.show()

plt.figure(figsize=(6, 4))
plt.imshow(S_q_mean, aspect="auto", interpolation="None")
plt.colorbar()
plt.show()

plt.figure(figsize=(6, 4))
plt.imshow(X_q_mean, aspect="auto", interpolation="None")
plt.colorbar()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np

import scipy
from scipy import special, spatial, sparse
import scipy.sparse.linalg

import matplotlib.pyplot as plt
from matplotlib import animation, rc, ticker
from mpl_toolkits.mplot3d import Axes3D
rc('animation', html='jshtml')

from IPython.display import clear_output, HTML

import math
import random
import time

from numba import jit, njit, prange
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
par_range = np.linspace(0, 1, 50)
q_vals = np.linspace(-5, 5, 100)[1:-1]

print(q_vals.shape, par_range.shape)
print(C_q_mean.shape)

xx, yy = np.meshgrid(q_vals, par_range)


fig = plt.figure(figsize=(12,10))
ax = plt.axes(projection='3d')
idk = ax.plot_surface(
    xx, yy, np.exp(C_q_mean), cmap="hsv", rstride=2, cstride=2, 
    shade=False, linewidth=0.05, antialiased=True, edgecolor="black", 
    label="String", vmin=1, vmax=1.07)
# idk._edgecolors2d=idk._edgecolors3d  # fixes some weird bug when using ax.legend()
# idk._facecolors2d=idk._facecolors3d
# ax.plot_wireframe(xx, yy, m, cmap = 'coolwarm',  lw=1, rstride=1, cstride=1)
# ax.set_title('')
ax.set_xlabel('q_vals')
ax.set_ylabel('par_range')
ax.set_zlabel('Cq')
ax.view_init(25, 260)
# ax.legend()
# idk.set_clim(-1,1)
# fig.colorbar(idk, shrink=0.3, aspect=10, pad=0)
plt.tight_layout()
# fig.legend()
# plt.savefig("img/E", dpi=300)
plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
par_range = np.linspace(0, 1, 50)
q_vals = np.linspace(-5, 5, 100)[1:]
print(q_vals.shape, par_range.shape)
print(C_q_mean.shape)

xx, yy = np.meshgrid(q_vals, par_range)


fig = plt.figure(figsize=(12,10))
ax = plt.axes(projection='3d')
idk = ax.plot_surface(
    xx, yy, np.exp(S_q_mean), cmap="hsv", rstride=2, cstride=2, 
    shade=False, linewidth=0.05, antialiased=True, edgecolor="black", 
    label="String", vmin=0, vmax=1.07)
# idk._edgecolors2d=idk._edgecolors3d  # fixes some weird bug when using ax.legend()
# idk._facecolors2d=idk._facecolors3d
# ax.plot_wireframe(xx, yy, m, cmap = 'coolwarm',  lw=1, rstride=1, cstride=1)
# ax.set_title('')
ax.set_xlabel('q_vals')
ax.set_ylabel('par_range')
ax.set_zlabel('Cq')
ax.view_init(25, 245)
# ax.legend()
# idk.set_clim(-1,1)
# fig.colorbar(idk, shrink=0.3, aspect=10, pad=0)
plt.tight_layout()
# fig.legend()
# plt.savefig("img/E", dpi=300)
plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lags = np.arange(10,100,10)
# lags = [1] 

miu = np.zeros((len(lags),500))
for i, lag in enumerate(lags):
    miu_bc = S[lag:] - S[0:-lag]
    miu[i,:]= miu_bc[500-lag:]
    plt.plot(miu[i,:].T)
    plt.show()
#     miu[i,:]= miu_bc[200-lag:]

# plt.figure(figsize=(15,5))
# plt.plot(miu.T)
# plt.show()

# miu_cut = miu[::, 1800:2600]
# plt.figure(figsize=(15,5))
# plt.plot(np.mean(miu_cut, axis=0))
# plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# G - agents 
# P - portfolio
# N - net worth
# S - stock price
# X - log returns
# D - decision
# T - stack
# U - sum_called_shares
# C - margin calls

@njit(parallel=True)
def parallel_simulation(PAR_range, SIM=100):

    G_MEAN = np.zeros((len(PAR_range),SIM),dtype=np.float64)
    G_STD = np.zeros((len(PAR_range),SIM),dtype=np.float64)

    P_MEAN = np.zeros((len(PAR_range),SIM),dtype=np.float64)
    P_STD = np.zeros((len(PAR_range),SIM),dtype=np.float64)

    N_MEAN = np.zeros((len(PAR_range),SIM),dtype=np.float64)
    N_STD = np.zeros((len(PAR_range),SIM),dtype=np.float64)

    S_MEAN = np.zeros((len(PAR_range),SIM),dtype=np.float64)
    S_STD = np.zeros((len(PAR_range),SIM),dtype=np.float64)

    X_MEAN = np.zeros((len(PAR_range),SIM), dtype=np.float64)
    X_STD = np.zeros((len(PAR_range),SIM),dtype=np.float64)

    D_MEAN = np.zeros((len(PAR_range),SIM), dtype=np.float64)
    D_STD = np.zeros((len(PAR_range),SIM),dtype=np.float64)

    T_MEAN = np.zeros((len(PAR_range),SIM), dtype=np.float64)
    T_STD = np.zeros((len(PAR_range),SIM),dtype=np.float64)

    C_MEAN = np.zeros((len(PAR_range),SIM), dtype=np.float64)
    C_STD = np.zeros((len(PAR_range),SIM),dtype=np.float64)

    for i in prange(len(PAR_range)):
        PAR_VAL = PAR_range[i]
        for j in prange(SIM):
            G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = False, pd = 0.05, pe = 0.01,
                    ph = 0.0485, pa = 0.7, N0=1000, N1 = 100, A =PAR_VAL, a=1, h=1, 
                    pi1 = 0.5, pi2 = 0.3, pi3 = 0.2)

            G_MEAN[i,j] = np.mean(G)
            G_STD[i,j] = np.std(G)

            P_MEAN[i,j] = np.mean(P)
            P_STD[i,j] = np.std(P)

            N_MEAN[i,j] = np.mean(N)
            N_STD[i,j] = np.std(N)

            S_MEAN[i,j] = np.mean(S)
            S_STD[i,j] = np.std(S)

            X_MEAN[i,j] = np.mean(X)
            X_STD[i,j] = np.std(X)

            D_MEAN[i,j] = np.mean(D)
            D_STD[i,j] = np.std(D)

            T_MEAN[i,j] = np.mean(T)
            T_STD[i,j] = np.std(T)

            C_MEAN[i,j] = np.mean(C)
            C_STD[i,j] = np.std(C)

    return (G_MEAN,G_STD,P_MEAN,P_STD, N_MEAN, N_STD, S_MEAN, S_STD, X_MEAN, X_STD, D_MEAN, D_STD, T_MEAN, T_STD, C_MEAN, C_STD)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PAR_range = np.linspace(0,16,100)
SIM = 1000

G_MEAN, G_STD, P_MEAN, P_STD, N_MEAN, N_STD, S_MEAN, S_STD, X_MEAN, X_STD, D_MEAN, D_STD, T_MEAN, T_STD, C_MEAN, C_STD = parallel_simulation(PAR_range, SIM)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# SIM 1 ---Bounded---
#             G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = True, pd = 0.05, pe = 0.01,
                    # ph = 0.0485, pa = 0.7, N0=1000, N1 = 100, A =PAR_VAL, a=1, h=1, 
                    # pi1 = 0.5, pi2 = 0.3, pi3 = 0.2) 

# PAR_range = np.linspace(0,16,100)

# SIM 2 ---NOTBounded---
#             G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = False, pd = 0.05, pe = 0.01,
                    # ph = 0.0485, pa = 0.7, N0=1000, N1 = 100, A =PAR_VAL, a=1, h=1, 
                    # pi1 = 0.5, pi2 = 0.3, pi3 = 0.2) 

PAR_range = np.linspace(0,16,100)

G_MEAN_MEAN =np.load("../../data/SIM1/G_MEAN_MEA_sim1000_A_var.npy")
P_MEAN_MEAN =np.load("../../data/SIM1/P_MEAN_MEA_sim1000_A_var.npy")
N_MEAN_MEAN =np.load("../../data/SIM1/N_MEAN_MEA_sim1000_A_var.npy")
S_MEAN_MEAN =np.load("../../data/SIM1/S_MEAN_MEA_sim1000_A_var.npy")
X_MEAN_MEAN =np.load("../../data/SIM1/X_MEAN_MEA_sim1000_A_var.npy")
D_MEAN_MEAN =np.load("../../data/SIM1/D_MEAN_MEA_sim1000_A_var.npy")
T_MEAN_MEAN =np.load("../../data/SIM1/T_MEAN_MEA_sim1000_A_var.npy")
C_MEAN_MEAN =np.load("../../data/SIM1/C_MEAN_MEA_sim1000_A_var.npy")
G_MEAN_STD  =np.load("../../data/SIM1/G_MEAN_STD_sim1000_A_var.npy")
P_MEAN_STD  =np.load("../../data/SIM1/P_MEAN_STD_sim1000_A_var.npy")
N_MEAN_STD  =np.load("../../data/SIM1/N_MEAN_STD_sim1000_A_var.npy")
S_MEAN_STD  =np.load("../../data/SIM1/S_MEAN_STD_sim1000_A_var.npy")
X_MEAN_STD  =np.load("../../data/SIM1/X_MEAN_STD_sim1000_A_var.npy")
D_MEAN_STD  =np.load("../../data/SIM1/D_MEAN_STD_sim1000_A_var.npy")
T_MEAN_STD  =np.load("../../data/SIM1/T_MEAN_STD_sim1000_A_var.npy")
C_MEAN_STD  =np.load("../../data/SIM1/C_MEAN_STD_sim1000_A_var.npy")
G_STD_MEAN  =np.load("../../data/SIM1/G_STD_MEAN_sim1000_A_var.npy")
P_STD_MEAN  =np.load("../../data/SIM1/P_STD_MEAN_sim1000_A_var.npy")
N_STD_MEAN  =np.load("../../data/SIM1/N_STD_MEAN_sim1000_A_var.npy")
S_STD_MEAN  =np.load("../../data/SIM1/S_STD_MEAN_sim1000_A_var.npy")
X_STD_MEAN  =np.load("../../data/SIM1/X_STD_MEAN_sim1000_A_var.npy")
D_STD_MEAN  =np.load("../../data/SIM1/D_STD_MEAN_sim1000_A_var.npy")
T_STD_MEAN  =np.load("../../data/SIM1/T_STD_MEAN_sim1000_A_var.npy")
C_STD_MEAN  =np.load("../../data/SIM1/C_STD_MEAN_sim1000_A_var.npy")
G_STD_STD   =np.load("../../data/SIM1/G_STD_STD _sim1000_A_var.npy")
P_STD_STD   =np.load("../../data/SIM1/P_STD_STD _sim1000_A_var.npy")
N_STD_STD   =np.load("../../data/SIM1/N_STD_STD _sim1000_A_var.npy")
S_STD_STD   =np.load("../../data/SIM1/S_STD_STD _sim1000_A_var.npy")
X_STD_STD   =np.load("../../data/SIM1/X_STD_STD _sim1000_A_var.npy")
D_STD_STD   =np.load("../../data/SIM1/D_STD_STD _sim1000_A_var.npy")
T_STD_STD   =np.load("../../data/SIM1/T_STD_STD _sim1000_A_var.npy")
C_STD_STD   =np.load("../../data/SIM1/C_STD_STD _sim1000_A_var.npy")
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

G_MEAN_MEAN = np.mean(G_MEAN, axis = 1)
P_MEAN_MEAN = np.mean(P_MEAN, axis = 1)
N_MEAN_MEAN = np.mean(N_MEAN, axis = 1) 
S_MEAN_MEAN = np.mean(S_MEAN, axis = 1)
X_MEAN_MEAN = np.mean(X_MEAN, axis = 1)
D_MEAN_MEAN = np.mean(D_MEAN, axis = 1)
T_MEAN_MEAN = np.mean(T_MEAN, axis = 1) 
C_MEAN_MEAN = np.mean(C_MEAN, axis = 1) 

G_MEAN_STD = np.std(G_MEAN, axis = 1)
P_MEAN_STD = np.std(P_MEAN, axis = 1)
N_MEAN_STD = np.std(N_MEAN, axis = 1) 
S_MEAN_STD = np.std(S_MEAN, axis = 1)
X_MEAN_STD = np.std(X_MEAN, axis = 1)
D_MEAN_STD = np.std(D_MEAN, axis = 1)
T_MEAN_STD = np.std(T_MEAN, axis = 1) 
C_MEAN_STD = np.std(C_MEAN, axis = 1) 

G_STD_MEAN = np.mean(G_STD, axis = 1)
P_STD_MEAN = np.mean(P_STD, axis = 1) 
N_STD_MEAN = np.mean(N_STD, axis = 1) 
S_STD_MEAN = np.mean(S_STD, axis = 1) 
X_STD_MEAN = np.mean(X_STD, axis = 1) 
D_STD_MEAN = np.mean(D_STD, axis = 1) 
T_STD_MEAN = np.mean(T_STD, axis = 1) 
C_STD_MEAN = np.mean(C_STD, axis = 1)

G_STD_STD = np.std(G_STD, axis = 1)
P_STD_STD = np.std(P_STD, axis = 1) 
N_STD_STD = np.std(N_STD, axis = 1) 
S_STD_STD = np.std(S_STD, axis = 1) 
X_STD_STD = np.std(X_STD, axis = 1) 
D_STD_STD = np.std(D_STD, axis = 1) 
T_STD_STD = np.std(T_STD, axis = 1) 
C_STD_STD = np.std(C_STD, axis = 1)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plt.figure(figsize=(15,5))
plt.ylabel("G_MEAN_MEAN")
plt.xlabel("A parameter values")
plt.grid()
plt.plot(PAR_range, G_MEAN_MEAN, c="r")
plt.fill_between(x=PAR_range, y1 =G_MEAN_MEAN+1.96*G_MEAN_STD/(np.sqrt(1000)), y2 = G_MEAN_MEAN-1.96*G_MEAN_STD/(np.sqrt(1000)), alpha =0.4 )
plt.show()

plt.figure(figsize=(15,5))
plt.ylabel("G_STD_MEAN")
plt.xlabel("A parameter values")
plt.grid()
plt.plot(PAR_range, G_STD_MEAN, c="r")
plt.fill_between(x=PAR_range, y1 =G_STD_MEAN+1.96*G_STD_STD/(np.sqrt(1000)), y2 = G_STD_MEAN-1.96*G_STD_STD/(np.sqrt(1000)), alpha =0.4 )
plt.show()

plt.figure(figsize=(15,5))
plt.ylabel("P_MEAN_MEAN")
plt.xlabel("A parameter values")
plt.grid()
plt.plot(PAR_range, P_MEAN_MEAN, c="r")
plt.fill_between(x=PAR_range, y1 =P_MEAN_MEAN+1.96*P_MEAN_STD/(np.sqrt(1000)), y2 = P_MEAN_MEAN-1.96*P_MEAN_STD/(np.sqrt(1000)), alpha =0.4 )
plt.show()

plt.figure(figsize=(15,5))
plt.ylabel("P_STD_MEAN")
plt.xlabel("A parameter values")
plt.grid()
plt.plot(PAR_range, P_STD_MEAN, c="r")
plt.fill_between(x=PAR_range, y1 =P_STD_MEAN+1.96*P_STD_STD/(np.sqrt(1000)), y2 = P_STD_MEAN-1.96*P_STD_STD/(np.sqrt(1000)), alpha =0.4 )
plt.show()

plt.figure(figsize=(15,5))
plt.ylabel("N_MEAN_MEAN")
plt.xlabel("A parameter values")
plt.grid()
plt.plot(PAR_range, N_MEAN_MEAN, c="r")
plt.fill_between(x=PAR_range, y1 =N_MEAN_MEAN+1.96*N_MEAN_STD/(np.sqrt(1000)), y2 = N_MEAN_MEAN-1.96*N_MEAN_STD/(np.sqrt(1000)), alpha =0.4 )
plt.show()

plt.figure(figsize=(15,5))
plt.ylabel("N_STD_MEAN")
plt.xlabel("A parameter values")
plt.grid()
plt.plot(PAR_range, P_MEAN_MEAN, c="r")
plt.fill_between(x=PAR_range, y1 =P_MEAN_MEAN+1.96*P_MEAN_STD/(np.sqrt(1000)), y2 = P_MEAN_MEAN-1.96*P_MEAN_STD/(np.sqrt(1000)), alpha =0.4 )
plt.show()

plt.figure(figsize=(15,5))
plt.ylabel("S_MEAN_MEAN")
plt.xlabel("A parameter values")
plt.grid()
plt.plot(PAR_range, S_MEAN_MEAN, c= "r")
plt.fill_between(x=PAR_range, y1 =S_MEAN_MEAN+1.96*S_MEAN_STD/(np.sqrt(1000)), y2 = S_MEAN_MEAN-1.96*S_MEAN_STD/(np.sqrt(1000)), alpha =0.4 )
plt.show()

plt.figure(figsize=(15,5))
plt.ylabel("S_STD_MEAN")
plt.xlabel("A parameter values")
plt.grid()
plt.plot(PAR_range, S_STD_MEAN, c="r")
plt.fill_between(x=PAR_range, y1 =S_STD_MEAN+1.96*S_STD_STD/(np.sqrt(1000)), y2 = S_STD_MEAN-1.96*S_STD_STD/(np.sqrt(1000)), alpha =0.4 )
plt.show()

plt.figure(figsize=(15,5))
plt.ylabel("X_MEAN_MEAN")
plt.xlabel("A parameter values")
plt.grid()
plt.plot(PAR_range, X_MEAN_MEAN, c="r")
plt.fill_between(x=PAR_range, y1 =X_MEAN_MEAN+1.96*X_MEAN_STD/(np.sqrt(1000)), y2 = X_MEAN_MEAN-1.96*X_MEAN_STD/(np.sqrt(1000)), alpha =0.4 )
plt.show()

plt.figure(figsize=(15,5))
plt.ylabel("X_STD_MEAN")
plt.xlabel("A parameter values")
plt.grid()
plt.plot(PAR_range, X_STD_MEAN, c="r")
plt.fill_between(x=PAR_range, y1 =X_STD_MEAN+1.96*X_STD_STD/(np.sqrt(1000)), y2 = X_STD_MEAN-1.96*X_STD_STD/(np.sqrt(1000)), alpha =0.4 )
plt.show()

plt.figure(figsize=(15,5))
plt.ylabel("D_MEAN_MEAN")
plt.xlabel("A parameter values")
plt.grid()
plt.plot(PAR_range, D_MEAN_MEAN, c="r")
plt.fill_between(x=PAR_range, y1 =D_MEAN_MEAN+1.96*D_MEAN_STD/(np.sqrt(1000)), y2 = D_MEAN_MEAN-1.96*D_MEAN_STD/(np.sqrt(1000)), alpha =0.4 )
plt.show()

plt.figure(figsize=(15,5))
plt.ylabel("D_STD_MEAN")
plt.xlabel("A parameter values")
plt.grid()
plt.plot(PAR_range, D_STD_MEAN, c="r")
plt.fill_between(x=PAR_range, y1 =D_STD_MEAN+1.96*D_STD_STD/(np.sqrt(1000)), y2 = D_STD_MEAN-1.96*D_STD_STD/(np.sqrt(1000)), alpha =0.4 )
plt.show()

plt.figure(figsize=(15,5))
plt.ylabel("T_MEAN_MEAN")
plt.xlabel("A parameter values")
plt.grid()
plt.plot(PAR_range, T_MEAN_MEAN, c="r")
plt.fill_between(x=PAR_range, y1 =T_MEAN_MEAN+1.96*T_MEAN_STD/(np.sqrt(1000)), y2 = T_MEAN_MEAN-1.96*T_MEAN_STD/(np.sqrt(1000)), alpha =0.4 )
plt.show()

plt.figure(figsize=(15,5))
plt.ylabel("T_STD_MEAN")
plt.xlabel("A parameter values")
plt.grid()
plt.plot(PAR_range, T_STD_MEAN, c="r")
plt.fill_between(x=PAR_range, y1 =T_STD_MEAN+1.96*T_STD_STD/(np.sqrt(1000)), y2 = T_STD_MEAN-1.96*T_STD_STD/(np.sqrt(1000)), alpha =0.4 )
plt.show()

plt.figure(figsize=(15,5))
plt.ylabel("C_MEAN_MEAN")
plt.xlabel("A parameter values")
plt.grid()
plt.plot(PAR_range,C_MEAN_MEAN, c="r")
plt.fill_between(x=PAR_range, y1 =C_MEAN_MEAN+1.96*C_MEAN_STD/(np.sqrt(1000)), y2 = C_MEAN_MEAN-1.96*C_MEAN_STD/(np.sqrt(1000)), alpha =0.4 )
plt.show()

plt.figure(figsize=(15,5))
plt.ylabel("C_STD_MEAN")
plt.xlabel("A parameter values")
plt.grid()
plt.plot(PAR_range, C_STD_MEAN, c="r")
plt.fill_between(x=PAR_range, y1 =C_STD_MEAN+1.96*C_STD_STD/(np.sqrt(1000)), y2 = C_STD_MEAN-1.96*C_STD_STD/(np.sqrt(1000)), alpha =0.4 )
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# G_MEAN_MEAN =np.save("../../data/SIM2/G_MEAN_MEA_sim1000_A_var_nobounds", G_MEAN_MEAN)
# P_MEAN_MEAN =np.save("../../data/SIM2/P_MEAN_MEA_sim1000_A_var_nobounds", P_MEAN_MEAN)
# N_MEAN_MEAN =np.save("../../data/SIM2/N_MEAN_MEA_sim1000_A_var_nobounds", N_MEAN_MEAN)
# S_MEAN_MEAN =np.save("../../data/SIM2/S_MEAN_MEA_sim1000_A_var_nobounds", S_MEAN_MEAN)
# X_MEAN_MEAN =np.save("../../data/SIM2/X_MEAN_MEA_sim1000_A_var_nobounds", X_MEAN_MEAN)
# D_MEAN_MEAN =np.save("../../data/SIM2/D_MEAN_MEA_sim1000_A_var_nobounds", D_MEAN_MEAN)
# T_MEAN_MEAN =np.save("../../data/SIM2/T_MEAN_MEA_sim1000_A_var_nobounds", T_MEAN_MEAN)
# C_MEAN_MEAN =np.save("../../data/SIM2/C_MEAN_MEA_sim1000_A_var_nobounds", C_MEAN_MEAN)
# G_MEAN_STD  =np.save("../../data/SIM2/G_MEAN_STD_sim1000_A_var_nobounds", G_MEAN_STD )
# P_MEAN_STD  =np.save("../../data/SIM2/P_MEAN_STD_sim1000_A_var_nobounds", P_MEAN_STD )
# N_MEAN_STD  =np.save("../../data/SIM2/N_MEAN_STD_sim1000_A_var_nobounds", N_MEAN_STD )
# S_MEAN_STD  =np.save("../../data/SIM2/S_MEAN_STD_sim1000_A_var_nobounds", S_MEAN_STD )
# X_MEAN_STD  =np.save("../../data/SIM2/X_MEAN_STD_sim1000_A_var_nobounds", X_MEAN_STD )
# D_MEAN_STD  =np.save("../../data/SIM2/D_MEAN_STD_sim1000_A_var_nobounds", D_MEAN_STD )
# T_MEAN_STD  =np.save("../../data/SIM2/T_MEAN_STD_sim1000_A_var_nobounds", T_MEAN_STD )
# C_MEAN_STD  =np.save("../../data/SIM2/C_MEAN_STD_sim1000_A_var_nobounds", C_MEAN_STD )
# G_STD_MEAN  =np.save("../../data/SIM2/G_STD_MEAN_sim1000_A_var_nobounds", G_STD_MEAN )
# P_STD_MEAN  =np.save("../../data/SIM2/P_STD_MEAN_sim1000_A_var_nobounds", P_STD_MEAN )
# N_STD_MEAN  =np.save("../../data/SIM2/N_STD_MEAN_sim1000_A_var_nobounds", N_STD_MEAN )
# S_STD_MEAN  =np.save("../../data/SIM2/S_STD_MEAN_sim1000_A_var_nobounds", S_STD_MEAN )
# X_STD_MEAN  =np.save("../../data/SIM2/X_STD_MEAN_sim1000_A_var_nobounds", X_STD_MEAN )
# D_STD_MEAN  =np.save("../../data/SIM2/D_STD_MEAN_sim1000_A_var_nobounds", D_STD_MEAN )
# T_STD_MEAN  =np.save("../../data/SIM2/T_STD_MEAN_sim1000_A_var_nobounds", T_STD_MEAN )
# C_STD_MEAN  =np.save("../../data/SIM2/C_STD_MEAN_sim1000_A_var_nobounds", C_STD_MEAN )
# G_STD_STD   =np.save("../../data/SIM2/G_STD_STD _sim1000_A_var_nobounds", G_STD_STD  )
# P_STD_STD   =np.save("../../data/SIM2/P_STD_STD _sim1000_A_var_nobounds", P_STD_STD  )
# N_STD_STD   =np.save("../../data/SIM2/N_STD_STD _sim1000_A_var_nobounds", N_STD_STD  )
# S_STD_STD   =np.save("../../data/SIM2/S_STD_STD _sim1000_A_var_nobounds", S_STD_STD  )
# X_STD_STD   =np.save("../../data/SIM2/X_STD_STD _sim1000_A_var_nobounds", X_STD_STD  )
# D_STD_STD   =np.save("../../data/SIM2/D_STD_STD _sim1000_A_var_nobounds", D_STD_STD  )
# T_STD_STD   =np.save("../../data/SIM2/T_STD_STD _sim1000_A_var_nobounds", T_STD_STD  )
# C_STD_STD   =np.save("../../data/SIM2/C_STD_STD _sim1000_A_var_nobounds", C_STD_STD  )
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@njit(parallel=True)
def parallel_simulation_crashes(PAR_range, SIM, threshold):
    crashes = np.zeros((len(PAR_range), SIM), dtype=np.float64)

    for i in prange(len(PAR_range)):
        PAR_VAL = PAR_range[i]
        for j in prange(SIM):
            G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = False, pd = 0.05, pe = 0.01,
                    ph = 0.0485, pa = 0.7, N0=1000, N1 = 100, A =4, a=1, h=1, 
                    pi1 = 0.5, pi2 = 0.3, pi3 = 0.2)
        
            # CRASH DATA         
            condition = X < threshold
            crashes[i,j] = contiguous_regions(condition).shape[0]

    return crashes

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PAR_range = np.linspace(0.01,0.5,50)
SIM = 1000
threshold = -0.15

crashes = parallel_simulation_crashes(PAR_range, SIM, threshold)
crashes_mean = np.mean(crashes, axis=1)
crashes_std = np.std(crashes, axis=1)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig = plt.figure(figsize=(12, 8))
# plt.errorbar(PAR_range, np.mean(crashes, axis=1), yerr=np.std(crashes, axis=1), color="C4")
plt.plot(PAR_range, crashes_mean, c="r")
plt.fill_between(x=PAR_range, y1 =crashes_mean+1.96*crashes_std/(np.sqrt(1000)), y2 = crashes_mean-1.96*crashes_std/(np.sqrt(1000)), alpha =0.4 )

plt.grid(alpha=0.2)
plt.ylabel("Crashes")
plt.xlabel(r"$p_h$")
plt.show()
# %%
