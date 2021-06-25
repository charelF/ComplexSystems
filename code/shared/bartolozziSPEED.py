#%%

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from numba import jit

# np.random.seed(1)
# random.seed(1)

#%%


@jit(nopython=True)
def p(k, i, xi, A, a, h, k2coord, Gt):
    return 1 / (1 + math.exp(-2 * I(k, i, xi, A, a, h, k2coord, Gt)))

@jit(nopython=True)
def I(k, i, xi, A, a, h, k2coord, Gt):
    total = 0
    zeta = random.uniform(-1,1)  # sampled for each unique (k,i)
    for j in k2coord[k]:  # for each coordinate in cluster k
        eta = random.uniform(-1,1)  # different for each cell
        sigma = Gt[j]
        total += ((A*xi[k] + a*eta) * sigma)
    return ((1 / len(k2coord[k])) * total) + h*zeta

@jit(nopython=True)
def cluster_info(arr):
    """ number of clusters (nonzero fields separated by 0s) in array
        and size of cluster
    """
    data = []
    k2coord = []
    coord2k = np.empty_like(arr).astype(np.int64)
    k = -1
    new_cluster = True

    for i in range(0,len(arr)):

        if arr[i] == 0:
            new_cluster = True
            coord2k[i] = -1
        else:
            if new_cluster == True:
                k += 1
                k2coord.append([i])
                data.append(0)
            else:
                k2coord[k].append(i)
                data[k] += 1

            coord2k[i] = k
            new_cluster = False

    Ncl = len(data)  # number of clusters
    Nk = data  # Nk[k] = size of cluster k
    return Ncl, Nk, k2coord, coord2k

@jit(nopython=True)
def generate(pd, pe, ph, pa, N0, N1, A, a, h):

    G = np.zeros(shape=(N0,N1)).astype(np.int16)
    # G[0] = np.random.choice(a=[-1,0,1], p=[pa/2, 1-pa, pa/2], size=N1, replace=True)
    for i in range(N1):
        rn = random.random()
        if rn < pa/2:
            val = -1
        elif pa/2 < rn < pa:
            val = 1
        else:
            val = 0
        G[0,i] = val
    
    x = np.empty(N0)

    for t in range(N0):
        Ncl, Nk, k2coord, coord2k = cluster_info(G[t])

        xi = np.random.uniform(-1, 1, size=Ncl)  # unique xi for each cluster k
        
        choice = np.array([-1,1])

        xt = 0
        for k, size in enumerate(Nk):
            tmp = 0
            for i in k2coord[k]:
                tmp += G[t,i]
            xt += size * tmp
        x[t] = xt

        if t == N0-1:
            # last iteration, we stop
            break

        for i in range(N1):
            # traders update their stance
            if G[t,i] != 0:
                k = coord2k[i]
                # print(k)
                pp = p(k, i, xi, A, a, h, k2coord, G[t])
                if random.random() < pp:
                    G[t+1,i] = 1
                else:
                    G[t+1,i] = -1

            
            # trader influences non-active neighbour to join
            if G[t,i] != 0:
                stance = G[t,i]
                if random.random() < ph:
                    if G[t,(i-1)%N1] == 0 and G[t,(i+1)%N1] == 0:
                        ni = np.random.choice(choice)
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
                    G[t+1,i] = np.random.choice(choice)

    return G, x

# %%

if __name__ == "__main__":

    N0 = 500
    N1 = 100

    pd = 0.1
    pe = 0.0001
    ph = 0.1

    pa = 0.5

    A = 2
    a = 0.1
    h = 0.1

    G, x = generate(pd, pe, ph, pa, N0, N1, A, a, h)
    
    fig, (ax1, ax2) = plt.subplots(
        ncols=1, nrows=2, figsize=(12,5), sharex=True, gridspec_kw = {'wspace':0, 'hspace':0}
    )
    ax1.imshow(G.T, cmap="binary", interpolation="None", aspect="auto")
    # plt.colorbar()

    r = (x - np.mean(x)) / np.std(x)
    print(sum(r**2))

    s = 100
    S = np.zeros_like(x)
    S[0] = s
    for i in range(1,N0):
        # S[i] = S[i-1] + (S[i-1] * r[i])
        S[i] = S[i-1] + (S[i-1] * r[i]/100) + 0.01

    ax2.plot(S)
    ax2.grid(alpha=0.4)

    ax2.set_xlabel("time")
    ax2.set_ylabel("close price")
    ax1.set_ylabel("agents")

    plt.tight_layout()
    plt.show()



# %%
