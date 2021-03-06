#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


#%%

# pd = 0.25
# pe = 0.02
# ph = 0.18 # vary
pd = 0.1
pe = 0.0001
ph = 0.1 # vary

pa = 1

N0 = 500
N1 = 40

# A = 2
# a = 0.1
# h = 0.1
A = 1
a = 1
h = 1

G = np.zeros(shape=(N0,N1))
G[0] = np.random.choice(a=[-1,0,1], p=[pa/2, 1-pa, pa/2], size=N1, replace=True)

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
S_ma = np.zeros(N0)
S_ma[0] = initial_stock_price
# X[0] = 1

DRIFT = 0.5
MA_T = 50

p_ma = 0

for t in range(N0-1):
    Ncl, Nk, k2coord, coord2k = cluster_info(G[t])

    Xt = 0
    for k, size in enumerate(Nk):
        tmp = 0
        for i in k2coord[k]:
            tmp += G[t,i]
        Xt += size * tmp
    X[t+1] = Xt/(10*N0)
    S[t+1] = S[t]*math.exp(X[t]) + DRIFT
    S_ma[t+1] = np.mean(S[max(0, (t+1 - MA_T)):t+1])

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
            for j in k2coord[k]:  # for each coordinate in cluster k
                eta = random.uniform(-1,1)  # different for each cell
                sigma = G[t,j]
                cluster_influence = A*xi[k]
                member_influence = a*eta
                total += ((cluster_influence + member_influence) * sigma)
            self_influence = h*zeta

            

            # # perf = percentage increase or decrease (pos or neg val)
            # performance = (N[t,i] - initial_account_balance) / initial_account_balance
            # # strat in [-1,1], high --> prefers buying, low --> prefers selling
            # strategy = np.mean(P[:t+1,i])
            # # print(f"Perf {performance}, Strategy {strategy}")
            # # perf high --> continue with strat
            # # perf low --> change strat
            # # perf <0 --> switch sign of strat
            # bias = performance * strategy
            # normalised_bias = 2 / (1 + np.exp(-2 * bias)) - 1
            # self_influence = normalised_bias * h
            
            ## fair valuation 
            ma_diff = (S_ma[t+1] - S[t+1]) / (np.std(S[:t+1]))
            ma_diff_norm = 2 / (1 + np.exp(-2 * ma_diff)) - 1
            ##print(f"MA diff norm {ma_diff_norm}")

            I = (1 / len(k2coord[k])) * total + self_influence 
            if(np.random.uniform(0, 1, 1) < p_ma):
                I += ma_diff_norm

            I = (1 / len(k2coord[k])) * total + self_influence + ma_diff_norm
            ##print(f"Sum {(1 / len(k2coord[k])) * total}, self influence {self_influence}")
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
                    G[t+1,(i+ni)%N1] = stance#random.choice([-1,1])
                elif G[t,(i-1)%N1] == 0:
                    G[t+1,(i-1)%N1] = stance#random.choice([-1,1])
                elif G[t,(i+1)%N1] == 0:
                    G[t+1,(i+1)%N1] = stance#random.choice([-1,1])
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

final_trade = P[-1] * S[-1]
B[-1] += final_trade
N[-1] = B[-1]

fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(
    ncols=1, nrows=5, figsize=(12,8), sharex=True, gridspec_kw = {'wspace':0, 'hspace':0.05}
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
cax2.get_xaxis().set_visible(False)
cax2.get_yaxis().set_visible(False)

cax3 = make_axes_locatable(ax3).append_axes('right', size=size, pad=0.05)
cax3.hist(X, orientation="horizontal", bins=np.linspace(np.min(X), np.max(X), len(X)//2))
cax3.get_xaxis().set_visible(False)
cax3.get_yaxis().set_visible(False)

# for ax in (ax2,ax3):
#     cax = make_axes_locatable(ax).append_axes('right', size=size, pad=0.05)
#     # cax.axis('off')

ax2.plot(S, label="Close Price")
ax2.plot(S_ma, label="MA")
ax2.grid(alpha=0.4)
ax2.legend()

ax3.bar(np.arange(len(X)), X)
ax3.grid(alpha=0.4)

ax4.set_xlabel("time")
# ax2.set_ylabel("standardised log returns")
ax2.set_ylabel("close price")
ax1.set_ylabel("agents")
ax3.set_ylabel("log return")
ax4.set_ylabel("portfolio")
ax5.set_ylabel("net worth")

# fig.colorbar(im, cax=ax4)

plt.tight_layout()
plt.show()

# %%

# %%
