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
        ph = 0.0485, pa = 0.7, N0=1000, N1 = 100, A =16, a=1, h=1, 
        pi1 = 0.5, pi2 = 0.3, pi3 = 0.2)

visualiseNICE(G,P,N,S,X,D,T,U,C)
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

# PAR_range = np.linspace(0,16,100)

# G_MEAN_MEAN =np.load("../../data/SIM1/G_MEAN_MEA_sim1000_A_var.npy")
# P_MEAN_MEAN =np.load("../../data/SIM1/P_MEAN_MEA_sim1000_A_var.npy")
# N_MEAN_MEAN =np.load("../../data/SIM1/N_MEAN_MEA_sim1000_A_var.npy")
# S_MEAN_MEAN =np.load("../../data/SIM1/S_MEAN_MEA_sim1000_A_var.npy")
# X_MEAN_MEAN =np.load("../../data/SIM1/X_MEAN_MEA_sim1000_A_var.npy")
# D_MEAN_MEAN =np.load("../../data/SIM1/D_MEAN_MEA_sim1000_A_var.npy")
# T_MEAN_MEAN =np.load("../../data/SIM1/T_MEAN_MEA_sim1000_A_var.npy")
# C_MEAN_MEAN =np.load("../../data/SIM1/C_MEAN_MEA_sim1000_A_var.npy")
# G_MEAN_STD  =np.load("../../data/SIM1/G_MEAN_STD_sim1000_A_var.npy")
# P_MEAN_STD  =np.load("../../data/SIM1/P_MEAN_STD_sim1000_A_var.npy")
# N_MEAN_STD  =np.load("../../data/SIM1/N_MEAN_STD_sim1000_A_var.npy")
# S_MEAN_STD  =np.load("../../data/SIM1/S_MEAN_STD_sim1000_A_var.npy")
# X_MEAN_STD  =np.load("../../data/SIM1/X_MEAN_STD_sim1000_A_var.npy")
# D_MEAN_STD  =np.load("../../data/SIM1/D_MEAN_STD_sim1000_A_var.npy")
# T_MEAN_STD  =np.load("../../data/SIM1/T_MEAN_STD_sim1000_A_var.npy")
# C_MEAN_STD  =np.load("../../data/SIM1/C_MEAN_STD_sim1000_A_var.npy")
# G_STD_MEAN  =np.load("../../data/SIM1/G_STD_MEAN_sim1000_A_var.npy")
# P_STD_MEAN  =np.load("../../data/SIM1/P_STD_MEAN_sim1000_A_var.npy")
# N_STD_MEAN  =np.load("../../data/SIM1/N_STD_MEAN_sim1000_A_var.npy")
# S_STD_MEAN  =np.load("../../data/SIM1/S_STD_MEAN_sim1000_A_var.npy")
# X_STD_MEAN  =np.load("../../data/SIM1/X_STD_MEAN_sim1000_A_var.npy")
# D_STD_MEAN  =np.load("../../data/SIM1/D_STD_MEAN_sim1000_A_var.npy")
# T_STD_MEAN  =np.load("../../data/SIM1/T_STD_MEAN_sim1000_A_var.npy")
# C_STD_MEAN  =np.load("../../data/SIM1/C_STD_MEAN_sim1000_A_var.npy")
# G_STD_STD   =np.load("../../data/SIM1/G_STD_STD _sim1000_A_var.npy")
# P_STD_STD   =np.load("../../data/SIM1/P_STD_STD _sim1000_A_var.npy")
# N_STD_STD   =np.load("../../data/SIM1/N_STD_STD _sim1000_A_var.npy")
# S_STD_STD   =np.load("../../data/SIM1/S_STD_STD _sim1000_A_var.npy")
# X_STD_STD   =np.load("../../data/SIM1/X_STD_STD _sim1000_A_var.npy")
# D_STD_STD   =np.load("../../data/SIM1/D_STD_STD _sim1000_A_var.npy")
# T_STD_STD   =np.load("../../data/SIM1/T_STD_STD _sim1000_A_var.npy")
# C_STD_STD   =np.load("../../data/SIM1/C_STD_STD _sim1000_A_var.npy")
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
# %%
