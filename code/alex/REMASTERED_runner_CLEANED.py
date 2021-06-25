#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit, prange

import matplotlib.pyplot as plt
from matplotlib import  rc
from mpl_toolkits.mplot3d import Axes3D
rc('animation', html='jshtml')

from numba import jit, njit, prange

import sys
sys.path.append('../shared')

from wednesdaySPEED import *
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        period = X[i:i+window]+1
        prod = np.prod(period)
        geo_mean = prod ** (1/window)
        if geo_mean < treshold:
            crashes += 1

    return crashes

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def visualiseNICE(G, P, N, S, X, D, T, U, C):
    """
    Generate a nice visualization of all raw data collected from the simulation
    - G: matrix N1xN0 containing the activity of agents (+ buy) (- sell) (0 inactive)
    - P: matrix N1xN0 portfolio of each agent overtime
    - N: matrix N1xN0 networth of each agent over time
    - S: array 1XN0 with prices of stock 
    - X: array 1XN0 with log returns of stock
    - D: matrix N1xN0 decision of each agent over time
    - T: stack
    - U: sum_called_shares
    - C: margin calls
    """
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

    ax2.plot(S, label="S")
    Ws = [25]
    for W in Ws:
        ax2.plot(np.arange(W-1, len(S)), moving_average(S, W), label=f"MA{W}")
    ax2.grid(alpha=0.4)

    ax3.bar(np.arange(len(X)), X)
    ax3.grid(alpha=0.4)

    ax6.plot(np.mean(D[0],axis=1), color="C0", alpha=1, label="CA")
    ax6.plot(np.mean(D[1],axis=1), color="C1", alpha=1, label="momentum")
    ax6.plot(np.mean(D[2],axis=1), color="C2", alpha=1, label="invert")
    ax6.plot(np.max(D[0],axis=1), ":", color="C0", alpha=1, label="CA")
    ax6.plot(np.max(D[1],axis=1), ":", color="C1", alpha=1, label="momentum")
    ax6.plot(np.max(D[2],axis=1), ":", color="C2", alpha=1, label="invert")
    ax6.plot(np.min(D[0],axis=1), "--", color="C0", alpha=1, label="CA")
    ax6.plot(np.min(D[1],axis=1), "--", color="C1", alpha=1, label="momentum")
    ax6.plot(np.min(D[2],axis=1), "--", color="C2", alpha=1, label="invert")
    ax6.grid(alpha=0.4)

    ax7.set_yscale("symlog")
    ax7.plot(T, label="stack")
    ax7.plot(U, label="called shares")
    ax7.grid(alpha=0.4)
    ax7.legend()

    ax8.imshow(C.T, cmap="binary", interpolation="None", aspect="auto")

    ax8.set_xlabel("time")
    ax2.set_ylabel("close price")
    ax1.set_ylabel("agents")
    ax3.set_ylabel("log return")
    ax4.set_ylabel("portfolio")
    ax5.set_ylabel("net worth")
    ax6.set_ylabel("influence (I)")
    ax7.set_ylabel("stack")
    ax8.set_ylabel("margin calls")

    plt.tight_layout()
    plt.show()


@njit(parallel=True)
def parallel_simulation_phase_transition(PAR1_range, PAR2_range, PAR3_range, SIM, treshold, N0):
    """
    parallelize simulation for varying parameter values -- collect number of crashes (crashes) and stock price
    array
    - PAR1_range: range for pi1 - proportion of stochastic agents
    - PAR2_range: range for pi2 - proportion of deterministic momentum agents
    - PAR3_range: range for pi3 - proportion of deterministic moving average agents
    - SIM: number of simulations per paramater value
    - treshold: the log return that defines a crash: 
        - e.g. if 20% drop over 5 days = crash then the treshold should be 0.8
    - N0: length of the simulation in iterations.
    """
    crashes = np.zeros((len(PAR1_range), SIM), dtype=np.float64)
    S_arrays = np.zeros((len(PAR1_range), SIM, N0), dtype=np.float64)

    for i in prange(len(PAR1_range)):
        PAR1_VAL = PAR1_range[i]
        PAR2_VAL = PAR2_range[i]
        PAR3_VAL = PAR3_range[i]
        for j in prange(SIM):
            G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = False, pd = 0.05, pe = 0.01, ph = 0.0485, pa = 0.7, N0=N0,N1 = 100, A =4, a=1, h=1, pi1 = PAR1_VAL, pi2 = PAR2_VAL, pi3 = PAR3_VAL)   

            crashes[i,j] = count_crashes(X, treshold, window=5)
            S_arrays[i,j] = S

    return (crashes, S_arrays)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# PAR1 - pi1 changes investigation
# Parameter ranges - where we keep the ratio of PAR2 and PAR3 constant
PAR1_range = np.linspace(0, 1 ,50)
PAR2_range = (1-PAR1_range)*0.3 
PAR3_range = (1-PAR1_range)*0.2

SIM = 10
treshold = 0.8
N0 = 1000
crashes, S_ARRAY = parallel_simulation_phase_transition(PAR1_range,PAR2_range, PAR3_range, SIM, treshold, N0)

crashes_mean = np.mean(crashes, axis=1)
crashes_std = np.std(crashes, axis=1)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tau = 2 
N = 100 # size of segments of the time series
q_vals_granulization = 100
q_vals = np.linspace(-5, 5, q_vals_granulization)

C_q = np.zeros(q_vals.shape[0] - 2) 
X_q = np.zeros(q_vals.shape[0])
S_q = np.zeros(q_vals.shape[0] - 1)

lhs = np.zeros((q_vals.shape[0]))
rhs = np.zeros((q_vals.shape[0]))

C_q_collector = np.empty((len(PAR1_range), SIM, *C_q.shape))
X_q_collector = np.empty((len(PAR1_range), SIM, *X_q.shape))
S_q_collector = np.empty((len(PAR1_range), SIM, *S_q.shape))

for i_par,par in enumerate(PAR1_range):

    for sim in range(SIM):
        series = S_ARRAY[i_par, sim]
        splt = np.array_split(series, N)
        mu_i = np.zeros(len(splt))

        denom_sum = 0

        for i in range(len(splt)):
            denom_sum += np.abs(splt[i][tau] - splt[i][0])

        for j in range(len(splt)):
            mu_i[j] = np.abs(splt[j][tau] - splt[j][0]) / denom_sum


        for k, val in enumerate(q_vals):
            lhs[k] = np.log(np.sum(mu_i**val))
            rhs[k] = np.log(N)
            X_q[k] = lhs[k] / rhs[k]

        for l in range(1, q_vals.shape[0] - 1):
            C_q[l - 1] = X_q[l + 1] - 2 * X_q[l] + X_q[l - 1]
            S_q[l - 1] = X_q[l + 1] - X_q[l - 1]

        C_q_collector[i_par, sim] = C_q
        X_q_collector[i_par, sim] = X_q
        S_q_collector[i_par, sim] = S_q

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C_q_mean = np.nanmean(C_q_collector, axis=1)
X_q_mean = np.nanmean(X_q_collector, axis=1)
S_q_mean = np.nanmean(S_q_collector, axis=1)

# RAW PLOTS 
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
# 3D plot of specific heat capacity of the system over the pi1 x q parameter values space
par_range = PAR1_range
q_vals_1 = q_vals[1:-1]
xx, yy = np.meshgrid(q_vals_1, par_range)

fig = plt.figure(figsize=(20,20))
ax = plt.axes(projection='3d')
idk = ax.plot_surface(
    xx, yy, C_q_mean, cmap="turbo", rstride=2, cstride=2, 
    shade=False, linewidth=0.05, antialiased=True, edgecolor="black", 
    label="String", vmin=0, vmax=8*10**(-6))

ax.set_xlabel('q_vals')
ax.set_ylabel('par_range')
ax.set_zlabel('Cq')
ax.view_init(20,80)

plt.tight_layout()
plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 3D plot of entropy of the system over the pi1 x q parameter values space
q_vals_2 = q_vals[1:]
xx, yy = np.meshgrid(q_vals_2, par_range)

fig = plt.figure(figsize=(12,10))
ax = plt.axes(projection='3d')
idk = ax.plot_surface(
    xx, yy, S_q_mean, cmap="hsv", rstride=2, cstride=2, 
    shade=False, linewidth=0.05, antialiased=True, edgecolor="black", 
    label="String", vmin=-0.07, vmax=-0.01)

ax.set_xlabel('q_vals')
ax.set_ylabel('par_range')
ax.set_zlabel('Sq')
ax.view_init(10, 130)

plt.tight_layout()
plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 3D plot of free energy of the system over the pi1 x q parameter values space
xx, yy = np.meshgrid(q_vals, par_range)

fig = plt.figure(figsize=(6,5))
ax = plt.axes(projection='3d')
idk = ax.plot_surface(
    xx, yy, X_q_mean, cmap="summer", rstride=2, cstride=2, 
    shade=False, linewidth=0.05, antialiased=True, edgecolor="black", 
    label="String", vmin=0, vmax=3)
ax.set_xlabel('q_vals')
ax.set_ylabel('par_range')
ax.set_zlabel('Xq')
ax.view_init(20, 270)

plt.tight_layout()
plt.show()


# %%
