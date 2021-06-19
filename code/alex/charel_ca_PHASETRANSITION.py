#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import jit
import warnings
warnings.simplefilter("ignore")

np.random.seed(1)
random.seed(1)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def cluster_info(arr):
    """ number of clusters (nonzero fields separated by 0s) in array
        and size of cluster
    """
    data = []
    k2coord = {}
    k = 0
    if arr[0] != 0: # left boundary
        data.append(0) # we will increment later in loop  
        k2coord[k] = []
    else:
        k=-1

    # print("arr", arr)
    # print("data", data)
    
    for i in range(0,len(arr)-1):
        if arr[i] == 0 and arr[i+1] != 0:
            data.append(0)
            k += 1
            k2coord[k] = []
        if arr[i] != 0:
            data[-1] += 1
            k2coord[k].append(i)
    if arr[-1] != 0:
        if data:  # if array is not empty
            data[-1] += 1  # right boundary
            k2coord[k].append(len(arr)-1)
        else:
            data.append(1)  
            k2coord[k] = [len(arr)-1]
            
    Ncl = len(data)  # number of clusters
    Nk = data  # Nk[k] = size of cluster k
    coord2k = {e:k for k,v in k2coord.items() for e in v}
    return Ncl, Nk, k2coord, coord2k

def trunc(X, high, low):
    return min(high, max(X, low))
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

    # for ax in (ax2,ax3):
    #     cax = make_axes_locatable(ax).append_axes('right', size=size, pad=0.05)
    #     # cax.axis('off')

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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def simulation(trigger = False, bound = False, pd = 0.05, pe = 0.01,
               ph = 0.0485, pa = 0.7, N0=2000, N1 = 100, A =4, a=1, h=1, 
               pi1 = 0.5, pi2 = 0.3, pi3 = 0.2):

    MA10_AGENTS = 0.05   #  Required: MA2>MA4>MA10
    MA4_AGENTS = 0.3     # split between agents that use MA2,MA4 and MA10
    MA2_AGENTS = 0.6

    # max absolute influence
    max_I = 2

    initial_account_balance = 2000
    min_account_balance = 800
    initial_stock_price = 200

    drift = 0  # not really working
    max_look_back = 4
    max_treshold = 1

    # ==================================
    G = np.zeros(shape=(N0,N1))
    G[0] = np.random.choice(a=[-1,0,1], p=[pa/2, 1-pa, pa/2], size=N1, replace=True)
    # G[0] = ((np.arange(0,N1)*6//N1)%3)-1
    # G[0] = ((np.arange(0,N1)*1//N1)%3)-1

    P = np.zeros_like(G) # portfolio: number of stocks
    N = np.zeros_like(G) # Net worth
    B = np.zeros_like(G) # acc balance

    B[0] = initial_account_balance  # everyone start with 1000 money
    N[0] = B[0]  # noone has stock initially

    D = np.zeros((3, *G.shape))  # decision
    C = np.zeros_like(G)  # margin call

    X = np.zeros(N0)
    S = np.zeros(N0)
    S[0] = initial_stock_price

    # MA 
    S_ma2 = np.zeros(N0)
    S_ma2[0] = initial_stock_price
    S_ma4 = np.zeros(N0)
    S_ma4[0] = initial_stock_price
    S_ma10 = np.zeros(N0)
    S_ma10[0] = initial_stock_price
    MA_T_R= [2,4,10]

    # each of the N1 agents has different treshold
    treshold = np.random.random(size=N1)*max_treshold

    T = np.zeros(N0)
    U = np.zeros(N0)

    stack = 0
    max_to_be_sold = N1

    investor_type = np.random.choice(
        a=[0,1,2], size=N1, replace=True, p = [pi1,pi2,pi3])

    for t in range(N0-1):
        Ncl, Nk, k2coord, coord2k = cluster_info(G[t])

        T[t] = stack
        Xt = 0
        for k, size in enumerate(Nk):
            tmp = 0
            for i in k2coord[k]:
                tmp += G[t,i]
            Xt += size * tmp

        if abs(stack) > max_to_be_sold:
            to_be_sold = max_to_be_sold * (1 if stack > 0 else -1)
            stack -= to_be_sold
        else:
            to_be_sold = stack
            stack = 0

        Xt -= to_be_sold
        X[t+1] = Xt/(10*N0)
        S[t+1] = S[t]*math.exp(X[t]) + drift

        for MA_T in MA_T_R:
            if MA_T == MA_T_R[0]:
                S_ma2[t+1] = np.mean(S[max(0, (t+1 - MA_T)):t+1])
            elif MA_T == MA_T_R[1]:
                S_ma4[t+1] = np.mean(S[max(0, (t+1 - MA_T)):t+1])
            else:
                S_ma10[t+1] = np.mean(S[max(0, (t+1 - MA_T)):t+1])

        if t<2:
            ma_diff_norm2 = 0
            ma_diff_norm4 = 0
            ma_diff_norm10 = 0
        elif t<4:
            ma_diff2 = (S_ma2[t+1] - S[t+1]) / np.std(S[:t+1]) # negative if stock>MA
            ma_diff_norm2 = 2 / (1 + np.exp(-2 * ma_diff2)) - 1
            ma_diff_norm4 = 0
            ma_diff_norm10 = 0
        elif t<10:
            ma_diff2 = (S_ma2[t+1] - S[t+1]) / np.std(S[:t+1]) # negative if stock>MA
            ma_diff_norm2 = 2 / (1 + np.exp(-2 * ma_diff2)) - 1
            ma_diff4 = (S_ma4[t+1] - S[t+1]) / np.std(S[:t+1]) # negative if stock>MA
            ma_diff_norm4 = 2 / (1 + np.exp(-2 * ma_diff4)) - 1
            ma_diff_norm10 = 0
        else:
            ma_diff2 = (S_ma2[t+1] - S[t+1]) / np.std(S[:t+1]) # negative if stock>MA
            ma_diff_norm2 = 2 / (1 + np.exp(-2 * ma_diff2)) - 1
            ma_diff4 = (S_ma4[t+1] - S[t+1]) / np.std(S[:t+1]) # negative if stock>MA
            ma_diff_norm4 = 2 / (1 + np.exp(-2 * ma_diff4)) - 1
            ma_diff10 = (S_ma10[t+1] - S[t+1]) / np.std(S[:t+1]) # negative if stock>MA
            ma_diff_norm10 = 2 / (1 + np.exp(-2 * ma_diff10)) - 1
                            

        xi = np.random.uniform(-1, 1, size=Ncl)  # unique xi for each cluster k

        for i in range(N1):
            P[t+1,i] = P[t,i] + G[t,i]
            # their next balance is their current balance minus
            # their purchase (or sell) of stock at current price
            B[t+1,i] = B[t,i] - (G[t,i] * S[t])
            N[t+1,i] = B[t,i] + (P[t,i]*S[t])

            if G[t,i] != 0:
                k = coord2k[i]
                total = 0
                zeta = random.uniform(-1,1)  # sampled for each unique (k,i)
                change = (S[t] - initial_stock_price) / initial_stock_price

                if investor_type[i] == 0 or investor_type[i]==2:
                    for j in k2coord[k]:  # for each coordinate in cluster k
                        eta = random.uniform(-1,1)  # different for each cell
                        sigma = G[t,j]
                        cluster_influence = A*xi[k]
                        member_influence = a*eta
                        total += ((cluster_influence + member_influence) * sigma)
                    self_influence = h*zeta
                    I = (1 / len(k2coord[k])) * total + self_influence
                    p = 1 / (1 + math.exp(-2 * I)) 

                if investor_type[i] == 1:

                    performance = (N[t,i] - initial_account_balance) / initial_account_balance
                    lookback = min(t,max_look_back)
                    strategy = np.mean(G[t-lookback:t+1,i])
                    bias = performance * strategy

                    trimmed_bias = max(-10, min(10, bias))
                    normalised_bias = 2 / (1 + math.exp(-2 * trimmed_bias)) - 1
                    self_influence = normalised_bias * h #* zeta
                    I = (1 / len(k2coord[k])) * total + self_influence 
                    p = 1 / (1 + math.exp(-2 * I))

                if investor_type[i] == 2:
                    MA_AGENTS_DECISION_NOISE = 0.8
                    VAL_RAND = np.random.normal(0, MA_AGENTS_DECISION_NOISE)

                    RATIONAL_RANDOM = random.random()
                    
                    p2 = 1 / (1 + math.exp(- ma_diff_norm2+VAL_RAND))
                    p4 = 1 / (1 + math.exp(- ma_diff_norm4+VAL_RAND))
                    p10 = 1 / (1 + math.exp(- ma_diff_norm10+VAL_RAND))
                    

                    if RATIONAL_RANDOM <= MA10_AGENTS:
                        p=p10
                        I = ma_diff_norm10+VAL_RAND

                    elif RATIONAL_RANDOM <= MA4_AGENTS:
                        p=p4
                        I = ma_diff_norm4+VAL_RAND
                    
                    elif RATIONAL_RANDOM <= MA2_AGENTS:
                        p=p2
                        I = ma_diff_norm2+VAL_RAND

                    else:
                        p=0.5
                        I = 1

                # Introduce shock
                if ((trigger ==True) and (t >=300 and t< 301)):
                    p = 1
                    I = max_I
                
                # Support line 
                if ((bound ==True) and (S[t] <=50)):
                    p = 0.7
                    I = max_I
                
                # Resistance
                if ((bound ==True) and (S[t] >=800)):
                    p = 0.3
                    I = max_I
                
                if random.random() < p:
                    decision = trunc(round(I),max_I,1)
                else:
                    decision = trunc(-abs(round(I)),-1,-max_I)

                G[t+1,i] = decision
                D[investor_type[i], t,i] = decision

            # trader influences non-active neighbour to join
            if G[t,i] != 0:
                stance = G[t,i]
                if random.random() < ph:
                    if G[t,(i-1)%N1] == 0 and G[t,(i+1)%N1] == 0:
                        ni = np.random.choice([-1,1])
                        G[t+1,(i+ni)%N1] = np.random.choice([-1,1])
                    elif G[t,(i-1)%N1] == 0:
                        G[t+1,(i-1)%N1] = np.random.choice([-1,1])
                    elif G[t,(i+1)%N1] == 0:
                        G[t+1,(i+1)%N1] = np.random.choice([-1,1])
                    else:
                        continue

            # active trader diffuses if it has inactive neighbour
            # only happens at edge of cluster
            if G[t,i] != 0:
                if random.random() < pd:
                    if (G[t,(i-1)%N1] == 0) or (G[t,(i+1)%N1] == 0):
                        G[t+1,i] = 0
                    else:
                        continue

            # nontrader enters market
            if G[t,i] == 0:
                if random.random() < pe:
                    G[t+1,i] = np.random.choice([-1,1])

        # margin call
        margin_call = N[t] < min_account_balance
        C[t+1] = margin_call
        # those that are margin called become inactive
        G[t+1] = G[t+1] * np.logical_not(margin_call) # those that are not remain
        P[t+1] = P[t+1] * np.logical_not(margin_call) # those that are not keep their portfolio
        # those that are are given the initial money again to start again
        B[t+1] = (B[t+1] * np.logical_not(margin_call)) + (initial_account_balance * margin_call)
        # they are also given their initial networth
        N[t+1] = (N[t+1] * np.logical_not(margin_call)) + (initial_account_balance * margin_call)
        # before we move on, we look at shares of those margin called
        sum_called_shares = sum(P[t] * margin_call)
        sum_margin_called = sum(margin_call)
        # these shares are sold at current price
        U[t+1] = sum_called_shares
        stack += sum_called_shares * sum_margin_called


    final_trade = P[-1] * S[-1]
    B[-1] += final_trade
    N[-1] = B[-1]

    return (G,P,N,S,X,D,T,U,C, initial_account_balance)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# =================== Parameters ================

G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = True, bound = True, pd = 0.05, pe = 0.01,
               ph = 0.0485, pa = 0.7, N0=2000, N1 = 100, A =4, a=1, h=1, 
               pi1 = 0.5, pi2 = 0.3, pi3 = 0.2)

visualiseNICE(G,P,N,S,X,D,T,U,C)






#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import pandas as pds

df = pds.read_csv("../../data/all_world_indices_clean.csv")
df_spx = df[["Date", "SPX Index"]]
df_spx["Date"] = pds.to_datetime(df_spx["Date"], format='%d/%m/%Y')
df_spx = df_spx.sort_values(by="Date")
df_spx.reset_index(inplace=True)
series_array = np.array(df_spx["SPX Index"])
log_ret_dat = np.diff(np.log(series_array))
log_ret_dat_stan = (log_ret_dat - np.mean(log_ret_dat)) / np.std(log_ret_dat)

r = (X - np.mean(X)) / np.std(X)

fig = plt.figure(figsize=(8, 8))
plt.yscale("log")
plt.hist(r, alpha=0.2, bins=50, label="CA", density=True)
plt.hist(log_ret_dat_stan, bins=50, alpha=0.2, label="S&P500", density=True)
plt.title("Log Return Distribution")
plt.legend()
plt.grid(alpha=0.2)
plt.show()

## back calc'd log returns for CA
# fig = plt.figure(figsize=(8, 8))
# plt.hist(, alpha=0.2, bins=50, label="CA", density=True)
# plt.hist(log_ret_dat_stan, bins=50, alpha=0.2, label="S&P500", density=True)
# plt.title("Log Return Distribution")
# plt.legend()
# plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import scipy.stats as stats
import statsmodels.api as sm
x_eval = np.linspace(-3, 3, 50)

kde1 = stats.gaussian_kde(r)
plt.plot(x_eval, kde1(x_eval), color="C4", label="CA Returns")

kde2 = stats.gaussian_kde(log_ret_dat_stan)
plt.plot(x_eval, kde2(x_eval), color="C9", label="S&P Returns")

plt.grid(alpha=0.2)
plt.legend()
plt.xlabel("r")
plt.ylabel("Prob Density")
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import scipy.stats as stats
import statsmodels.api as sm
acf_x_price = sm.tsa.stattools.acf(r)
acf_sp_price = sm.tsa.stattools.acf(log_ret_dat_stan)
x = np.arange(acf_x_price.shape[0])

mean_sp = np.mean(acf_sp_price)
fig = plt.figure(figsize=(15, 5))
plt.plot(x, acf_x_price, label="S&P500 Returns")
plt.plot(x, acf_sp_price, label="CA Returns")
plt.xlabel("Lag")
plt.ylabel("Autocorrelations")
plt.grid(alpha=0.2)
plt.legend()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
acf_x_vol = sm.tsa.stattools.acf(np.abs(r))
acf_sp_vol = sm.tsa.stattools.acf(np.abs(log_ret_dat_stan))
x = np.arange(acf_x_vol.shape[0])

fig = plt.figure(figsize=(15, 5))
plt.plot(x, acf_x_vol, label="S&P500 Volatility")
plt.plot(x, acf_sp_vol, label="CA Volatility")
plt.xlabel("Lag")
plt.ylabel("Autocorrelations")
plt.grid(alpha=0.2)
plt.legend()
plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
