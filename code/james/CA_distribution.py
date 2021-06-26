#%%

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pds
import math
import random

import statsmodels.api as sm
import scipy.stats as stats


import warnings
warnings.simplefilter("ignore")

np.random.seed(1)
random.seed(1)

#%%

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

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w



#%%

pd = 0.1 # active becomes inactive
pe = 0.01 # probability of nontrader to enter the market
ph = 0.0685  # probability that an active trader can turn one of his inactive neighbots into active
pa = 0.5

N0 = 1000
N1 = 100

A = 1.8
a = 1
h = 1

G = np.zeros(shape=(N0,N1))
G[0] = np.random.choice(a=[-1,0,1], p=[pa/2, 1-pa, pa/2], size=N1, replace=True)

## Cell Rule Types 
## 3 = percolation, 4 = MA, 5=portfolio management
T = np.zeros(shape=(N0,N1))
T_temp = np.random.choice(a=[3, 4, 5], p=[0.5, 0.2, 0.3], size=N1, replace=True)
##T[0] = np.where(((G[0] == -1) | (G[0] == 1)), T_temp, 0)
T[:,:] = T_temp

initial_account_balance = 1000
initial_stock_price = 100
P = np.zeros_like(G) # portfolio: number of stocks
N = np.zeros_like(G) # Net worth
B = np.zeros_like(G) # account Balance
B[0] = initial_account_balance  # everyone start with 1000 money
N[0] = B[0]  # noone has stock initially

X = np.zeros(N0)
S = np.zeros(N0)
S[0] = initial_stock_price

S_ma10 = np.zeros(N0)
S_ma10[0] = initial_stock_price
S_ma20 = np.zeros(N0)
S_ma20[0] = initial_stock_price
S_ma30 = np.zeros(N0)
S_ma30[0] = initial_stock_price

MA_T_R= [4, 8, 16]

DRIFT = 0.1
MAXLOOKBACK = 4

# each of the N1 agents has different treshold
treshold = np.random.random(size=N1)*10

for t in range(N0-1):
    ## filter array before giving to cluster info such that only the 
    ## entries with the percolation rule have their value calculated
    G_t_filtered = np.where(T[t] == 3, G[t], 0)
    Ncl, Nk, k2coord, coord2k = cluster_info(G[t])

    Xt = 0
    for k, size in enumerate(Nk):
        tmp = 0
        for i in k2coord[k]:
            tmp += G[t,i]
        Xt += size * tmp
    X[t+1] = Xt/(10*N0)
    S[t+1] = S[t]*math.exp(X[t]) + DRIFT

    for MA_T in MA_T_R:
    
        if MA_T == 2:
            S_ma10[t+1] = np.mean(S[max(0, (t+1 - MA_T)):t+1])
        elif MA_T == 4:
            S_ma20[t+1] = np.mean(S[max(0, (t+1 - MA_T)):t+1])
        else:
            S_ma30[t+1] = np.mean(S[max(0, (t+1 - MA_T)):t+1])


    xi = np.random.uniform(-1, 1, size=Ncl)  # unique xi for each cluster k

    for i in range(N1):
        P[t+1,i] = P[t,i] + G[t,i]
        # their next balance is their current balance minus
        # their purchase (or sell) of stock at current price
        B[t+1,i] = B[t,i] - (G[t,i] * S[t])
        N[t+1,i] = B[t,i] + (P[t,i]*S[t])

        # traders update their stance
        if G[t,i] != 0:
            k = coord2k[i]
            total = 0
            zeta = random.uniform(-1,1)  # sampled for each unique (k,i)

            if T[t, i] == 3:
                for j in k2coord[k]:  # for each coordinate in cluster k
                    eta = random.uniform(-1,1)  # different for each cell
                    sigma = G[t,j]
                    cluster_influence = A*xi[k]
                    member_influence = a*eta
                    total += ((cluster_influence + member_influence) * sigma)
                self_influence = h*zeta

                
                I = (1 / len(k2coord[k])) * total + self_influence
                p = 1 / (1 + math.exp(-2 * I))

                if random.random() < p:
                    G[t+1,i] = 1
                else:
                    G[t+1,i] = -1

            elif T[t, i] == 4:
                ## fair valuation 
                ma_diff10 = (S_ma10[t+1] - S[t+1]) / np.std(S[:t+1]) # negative if stock>MA
                ma_diff_norm10 = 2 / (1 + np.exp(-2 * ma_diff10)) - 1
                ma_diff20 = (S_ma20[t+1] - S[t+1]) / np.std(S[:t+1]) # negative if stock>MA
                ma_diff_norm20 = 2 / (1 + np.exp(-2 * ma_diff20)) - 1
                ma_diff30 = (S_ma30[t+1] - S[t+1]) / np.std(S[:t+1]) # negative if stock>MA
                ma_diff_norm30 = 2 / (1 + np.exp(-2 * ma_diff30)) - 1
                
                # RANDOM AGENT
                AR_PROB = 0.5 # MA agent
                AR_PERFORMANCE = 0.9 # portfolio agent
    
                RANDOM_AGENT, RATIONAL_RANDOM = random.random(), random.random() 

                VAL_RAND = np.random.normal(0,0.05)

                p10 = 1 / (1 + math.exp(- ma_diff_norm10+VAL_RAND))
                p20 = 1 / (1 + math.exp(- ma_diff_norm20+VAL_RAND))
                p30 = 1 / (1 + math.exp(- ma_diff_norm30+VAL_RAND))
                
                if ((random.random() <= 0.8) and (G[t,i]==0)):
                    G[t+1,i] = 0

                if RATIONAL_RANDOM <= 0.05:
                    if random.random() <= p30:
                        G[t+1,i] = +1
                    else:
                        G[t+1,i] = -1
                elif RATIONAL_RANDOM <= 0.3:
                    if random.random() <= p20:
                        G[t+1,i] = +1
                    else:
                        G[t+1,i] = -1
                elif RATIONAL_RANDOM <= 0.6:
                    if random.random() <= p10:
                        G[t+1,i] = +1
                    else:
                        G[t+1,i] = -1

            elif T[t, i] == 5:
                # perf = percentage increase or decrease (pos or neg val)
                performance = (N[t,i] - initial_account_balance) / initial_account_balance
                # strat in [-1,1], high --> prefers buying, low --> prefers selling
                lookback = min(t,MAXLOOKBACK)
                strategy = np.mean(G[t-lookback:t+1,i])
                bias = performance * strategy
                trimmed_bias = max(-10, min(10, bias))
                normalised_bias = 2 / (1 + math.exp(-2 * trimmed_bias)) - 1
                self_influence = normalised_bias * h #* zeta
                I = (1 / len(k2coord[k])) * total + self_influence 

                p = 1 / (1 + math.exp(-2 * I))
                if random.random() < p:
                    G[t+1,i] = 1
                else:
                    G[t+1,i] = -1


        # trader influences non-active neighbour to join
        if G[t,i] != 0:
            stance = G[t,i]
            if random.random() < ph:
                if G[t,(i-1)%N1] == 0 and G[t,(i+1)%N1] == 0:
                    ni = np.random.choice([-1,1])
                    if T[t, i] == T[t,(i+ni)%N1]:
                        G[t+1,(i+ni)%N1] = stance
                elif G[t,(i-1)%N1] == 0:
                    if T[t, i] == T[t,(i-1)%N1]:
                        G[t+1,(i-1)%N1] = stance
                elif G[t,(i+1)%N1] == 0:
                    if T[t, i] == T[t,(i+1)%N1]:                    
                        G[t+1,(i+1)%N1] = stance
                else:
                    continue

        # active trader diffuses if it has inactive neighbour
        # only happens at edge of cluster
        if G[t,i] != 0 and random.random() < pd:
            if (G[t,(i-1)%N1] == 0) and (T[t,(i-1)%N1] == T[t, i]): 
                G[t+1,i] = 0
            elif(G[t,(i+1)%N1] == 0) and (T[t,(i+1)%N1] == T[t, i]):
                G[t+1,i] = 0
            else:
                continue

        # nontrader enters market
        if G[t,i] == 0 and random.random() < pe:
            G[t+1,i] = np.random.choice([-1,1])

final_trade = P[-1] * S[-1]
B[-1] += final_trade
N[-1] = B[-1]

fig, (ax1,ax2,ax3,ax4,ax5, ax6) = plt.subplots(
    ncols=1, nrows=6, figsize=(12,9), sharex=True, gridspec_kw = 
    {'wspace':0, 'hspace':0.05, 'height_ratios':[1,2,1,1,1,1]}
)
im1 = ax1.imshow(G.T, cmap="bone", interpolation="None", aspect="auto")
im4 = ax4.imshow(P.T, cmap="hot", interpolation="None", aspect="auto")
amnwc = np.max(np.abs(N-initial_account_balance))  # absolute max net worth change
vmin, vmax = initial_account_balance-amnwc, initial_account_balance+amnwc
im5 = ax5.imshow(N.T, cmap="bwr", interpolation="None", aspect="auto", vmin=vmin, vmax=vmax)
im6 = ax6.imshow(T.T, cmap="bone", interpolation="None", aspect="auto")

size = "15%"

cax1 = make_axes_locatable(ax1).append_axes('right', size=size, pad=0.05)
fig.colorbar(im1, cax=cax1, orientation='vertical')
cax4 = make_axes_locatable(ax4).append_axes('right', size=size, pad=0.05)
fig.colorbar(im4, cax=cax4, orientation='vertical')
cax5 = make_axes_locatable(ax5).append_axes('right', size=size, pad=0.05)
fig.colorbar(im5, cax=cax5, orientation='vertical')
cax6 = make_axes_locatable(ax6).append_axes('right', size=size, pad=0.05)
fig.colorbar(im6, cax=cax6, orientation='vertical')

cax2 = make_axes_locatable(ax2).append_axes('right', size=size, pad=0.05)
cax2.hist(S, orientation="horizontal", bins=np.linspace(np.min(S), np.max(S), len(S)//2))
cax2.get_xaxis().set_visible(False)
cax2.get_yaxis().set_visible(False)

cax3 = make_axes_locatable(ax3).append_axes('right', size=size, pad=0.05)
cax3.hist(X, orientation="horizontal", bins=np.linspace(np.min(X), np.max(X), len(X)//5))
cax3.get_xaxis().set_visible(False)
cax3.get_yaxis().set_visible(False)

# for ax in (ax2,ax3):
#     cax = make_axes_locatable(ax).append_axes('right', size=size, pad=0.05)
#     # cax.axis('off')


ax2.plot(S, label="S")
Ws = [10]
for W in Ws:
    ax2.plot(np.arange(W-1, len(S)), moving_average(S, W), label=f"MA{W}")
ax2.grid(alpha=0.4)
ax2.legend(ncol=len(Ws)+1)

ax3.bar(np.arange(len(X)), X)
ax3.grid(alpha=0.4)

ax4.set_xlabel("time")
# ax2.set_ylabel("standardised log returns")
ax2.set_ylabel("close price")
ax1.set_ylabel("agents")
ax3.set_ylabel("log return")
ax4.set_ylabel("portfolio")
ax5.set_ylabel("net worth")
ax6.set_ylabel("agent types")

# fig.colorbar(im, cax=ax4)

plt.tight_layout()
plt.show()

# %%

df = pds.read_csv("../../data/all_world_indices_clean.csv")
df_spx = df[["Date", "SPX Index"]]
df_spx["Date"] = pds.to_datetime(df_spx["Date"], format='%d/%m/%Y')
df_spx = df_spx.sort_values(by="Date")
df_spx.reset_index(inplace=True)
series_array = np.array(df_spx["SPX Index"])
log_ret_dat = np.diff(np.log(series_array))
log_ret_dat_stan = (log_ret_dat - np.mean(log_ret_dat)) / np.std(log_ret_dat)

r = (X - np.mean(X)) / np.std(X)

print(np.std(r))
print(np.std(log_ret_dat_stan))

fig = plt.figure(figsize=(8, 8))
plt.hist(r, alpha=0.4, bins=30, label="CA", density=True)
plt.hist(log_ret_dat_stan, bins=30, alpha=0.4, label="S&P500", density=True)
plt.yscale("log")
plt.title("Log Return Distribution - Standardised")
plt.legend()
plt.grid(alpha=0.2)
plt.show()

# %%


fig = plt.figure(figsize=(8, 8))
plt.hist(X, alpha=0.2, bins=50, label="CA", density=True)
plt.hist(log_ret_dat, bins=50, alpha=0.2, label="S&P500", density=True)
plt.title("Log Return Distribution - Unstandardised")
plt.yscale("log")
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

# %%

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

# %%

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


# %%
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
# %%
