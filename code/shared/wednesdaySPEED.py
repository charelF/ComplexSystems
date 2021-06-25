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
def cluster_info(arr):
    """ returns various information about the clusters at a given timepoint which correponds to 1D array arr.
    the function makes somewhat clumsy use of list instead of a neater implementation with dictionaries as
    numba does not support python dictionaries.

    input:
    - arr: the 1D array of cells

    output:
    - Ncl: int: number of clusters
    - Nk: list: at index k, this list contains the length of cluster k
    - k2coord: list: at index k, this list contains a list with all the 1D indices of nodes in this cluster
    - coord2k: list: at index i, this list contains the cluster to which the cell at coordinate i belongs to
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
def trunc(X, high, low):
    """truncates the value X to be within [high, low]"""
    return min(high, max(X, low))

@jit(nopython=True)
def simulation(trigger = False, bound = False, pd = 0.05, pe = 0.01,
    ph = 0.0485, pa = 0.7, N0=2000, N1 = 100, A =4, a=1, h=1, 
    pi1 = 0.5, pi2 = 0.3, pi3 = 0.2, ub=1000, lb=20
):
    """ runs our model

    input:
    - trigger: [experimental feature] whether crashes are triggered at specific time intervals. not used in final experiments
    - bound: [experimental feature] whether to bound the stock price to an interval of [lb, ub]. not used in simulations
    - pd: probability of trader at edge of cluster leaving
    - pe: probability of random inactive trader becoming active
    - ph: probability of trader convincing one of its inactive neighbours to join
    - pa: initial distribution of inactive and active traders
    - N0: amount of timesteps to simulate the model for (length of grid)
    - N1: amount of cells (width of grid)
    - A: cluster influence to stochastic cells
    - a: neighbour influence to stochastic cells
    - h: self influence to stochastic cells
    - pi1: probability of a cell being of a stochastic trader as proposed by bartolozzi
    - pi2: probability of a cell being a deterministic momentum traders
    - pi3: probability of a cell being a deterministic moving average trader
    - ub and lb: upper and lower bound for stock price, only used if bounds=True

    notes:
    - pi1 + pi2 + pi3 must be 1
    - various other parameters have been investigated but deemed not interesting enough to be used as input
        parameters and are instead hardcoded

    output: G,P,N,S,X,D,T,U,C
    - G: matrix (N1xN0) containing the activity of agents (+ buy) (- sell) (0 inactive)
    - P: matrix (N1xN0) portfolio of each agent overtime
    - N: matrix (N1xN0) net worth of each agent over time
    - S: array (N1xN0) with prices of stock 
    - X: array (N1xN0) with log returns of stock
    - D: matrix (N1xN0) decision of each agent over time
    - T: array (1xN0) stack of the margin calls (not much used)
    - U: array (1xN0) sum_called_shares (sum of shares being margin called) (not much used)
    - C: matrix (N1xN0) margin calls (which cell got a margin call when)

    """

    # which averages moving average traders use
    # required: ma10 < ma4 < ma2
    ma10 = 0.05
    ma4 = 0.3
    ma2 = 0.6
    ma_list = [2,4,10]

    # max absolute influence
    max_I = 2

    initial_account_balance = 2000
    min_account_balance = 800
    initial_stock_price = 200

    drift = 0  # risk free interest rate / drift of stuck - not fully implemented
    max_look_back = 4  # max amount of days that momentum traders look back at their performance

    # the choice array. Written like this for compatibility with numba
    choice = np.array([-1,1], dtype=np.int8)

    # we create the initial distribution of agents
    # basically the same as np.random.choice with the p parameter however this function
    # is not supported by numba so we had to reimplement it
    G = np.zeros(shape=(N0,N1), dtype=np.int32)
    for i in range(N1):
        rn = random.random()
        if rn < pa/2:
            val = -1
        elif pa/2 < rn < pa:
            val = 1
        else:
            val = 0
        G[0,i] = val

    # various data collection arrays, explained in function docstring
    P = np.zeros(shape=(N0,N1), dtype=np.int32) # portfolio: number of stocks
    N = np.zeros(shape=(N0,N1), dtype=np.float64) # Net worth
    B = np.zeros(shape=(N0,N1), dtype=np.float64) # acc balance
    B[0] = initial_account_balance
    N[0] = B[0]
    D = np.zeros((3, *G.shape), dtype=np.int8)  # decision
    C = np.zeros_like(G, dtype=np.bool_)  # margin call
    X = np.zeros(N0, dtype=np.float64)
    S = np.zeros(N0, dtype=np.float64)
    S[0] = initial_stock_price
    T = np.zeros(N0)
    U = np.zeros(N0)

    # arrays which contain the moving average of the stock price
    S_ma2 = np.zeros(shape=N0, dtype=np.float64)
    S_ma2[0] = initial_stock_price
    S_ma4 = np.zeros(shape=N0, dtype=np.float64)
    S_ma4[0] = initial_stock_price
    S_ma10 = np.zeros(shape=N0, dtype=np.float64)
    S_ma10[0] = initial_stock_price

    stack = 0
    max_to_be_sold = N1
    max_stack = 5*N1

    # the random distribution of the investor type
    investor_type = np.zeros(shape=N1, dtype=np.int8)
    for i in range(N1):
        rn = random.random()
        if 0 < rn <= pi1:
            val = 0
        elif pi1 < rn <= pi1+pi2:
            val = 1
        else:
            val = 2
        investor_type[i] = val

    # looping over all time steps
    for t in range(N0-1):
        # first we get the cluster info
        Ncl, Nk, k2coord, coord2k = cluster_info(G[t])

        # next we compute the log returns as proposed by bartolozzi et al
        stack = trunc(stack, max_stack, -max_stack)
        T[t] = stack 
        Xt = 0
        for k, size in enumerate(Nk):
            tmp = 0
            for i in k2coord[k]:
                tmp += G[t,i]
            Xt += size * tmp

        # we also compute the stack of shares from previous margin calls that still need to be sold
        if abs(stack) > max_to_be_sold:
            to_be_sold = max_to_be_sold * (1 if stack > 0 else -1)
            stack -= to_be_sold
        else:
            to_be_sold = stack
            stack = 0
        Xt -= to_be_sold

        # finally we get the stock price and log returns for next round
        X[t+1] = Xt/(10*N0)
        S[t+1] = S[t]*math.exp(X[t]) + drift

        # and can comput the moving average
        S_ma2[t+1] = np.mean(S[max(0, (t+1 - 2)):t+1])
        S_ma4[t+1] = np.mean(S[max(0, (t+1 - 4)):t+1])
        S_ma10[t+1] = np.mean(S[max(0, (t+1 - 10)):t+1])
        std = np.std(S[:t+1])
        # the code is a bit convoluted to the probelm of not being able to comput the mean average if
        # not enough data is present
        if t<2 or std == 0:
            ma_diff_norm2 = 0
            ma_diff_norm4 = 0
            ma_diff_norm10 = 0
        elif t<4:
            ma_diff2 = (S_ma2[t+1] - S[t+1]) / std # negative if stock>MA
            ma_diff_norm2 = 2 / (1 + np.exp(-2 * ma_diff2)) - 1
            ma_diff_norm4 = 0
            ma_diff_norm10 = 0
        elif t<10:
            ma_diff2 = (S_ma2[t+1] - S[t+1]) / std # negative if stock>MA
            ma_diff_norm2 = 2 / (1 + np.exp(-2 * ma_diff2)) - 1
            ma_diff4 = (S_ma4[t+1] - S[t+1]) / std # negative if stock>MA
            ma_diff_norm4 = 2 / (1 + np.exp(-2 * ma_diff4)) - 1
            ma_diff_norm10 = 0
        else:
            ma_diff2 = (S_ma2[t+1] - S[t+1]) / std # negative if stock>MA
            ma_diff_norm2 = 2 / (1 + np.exp(-2 * ma_diff2)) - 1
            ma_diff4 = (S_ma4[t+1] - S[t+1]) / std # negative if stock>MA
            ma_diff_norm4 = 2 / (1 + np.exp(-2 * ma_diff4)) - 1
            ma_diff10 = (S_ma10[t+1] - S[t+1]) / std # negative if stock>MA
            ma_diff_norm10 = 2 / (1 + np.exp(-2 * ma_diff10)) - 1

        # generating the first random variable               
        xi = np.random.uniform(-1, 1, size=Ncl)  # unique xi for each cluster k

        # now we loop over each agents
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

                # first we look at the random agents
                # how they work is more closely explained in the file original_implementation
                # and also in the paper by Bartolozzi et al.
                if investor_type[i] == 0:
                    for j in k2coord[k]:  # for each coordinate in cluster k
                        eta = random.uniform(-1,1)  # different for each cell
                        sigma = G[t,j]
                        cluster_influence = A*xi[k]
                        member_influence = a*eta
                        total += ((cluster_influence + member_influence) * sigma)
                    self_influence = h*zeta
                    I = (1 / len(k2coord[k])) * total + self_influence
                    p = 1 / (1 + math.exp(-2 * I)) 

                # next we look at the momentum investors.
                # these basically compute their performance by looking at their performance and how
                # it corresponded to with their previous actions, and then they may change their strategy
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

                # lastly we look at the moving average investors.
                # these look at the relation between the stock and its moving average
                # they sell if the stock goes to high and buy if the stock goes too low
                if investor_type[i] == 2:
                    
                    rv_normal_noise = np.random.normal(0, 0.8)
                    rv_uniform = random.random()
                    
                    if rv_uniform <= ma10:
                        p = 1 / (1 + math.exp(-ma_diff_norm10+rv_normal_noise))
                        I = ma_diff_norm10 + rv_normal_noise

                    elif rv_uniform <= ma4:
                        p = 1 / (1 + math.exp(-ma_diff_norm4+rv_normal_noise))
                        I = ma_diff_norm4 + rv_normal_noise
                    
                    elif rv_uniform <= ma2:
                        p = 1 / (1 + math.exp(-ma_diff_norm2+rv_normal_noise))
                        I = ma_diff_norm2 + rv_normal_noise

                    else:
                        p = 0.5
                        I = 1

                # if trigger was enabled, here we introduce an outside event triggering a price increase
                if ((trigger == True) and (t == 300)):
                    p = 1
                    I = max_I
                
                # if bound was enable, here we introduce buy or sell walls that push the price to remain in the bounds
                if ((bound ==True) and (S[t] <= lb)):
                    p = 0.7
                    I = max_I
                if ((bound ==True) and (S[t] >= ub)):
                    p = 0.3
                    I = max_I
                
                # lastly, since the CA is stochastic, there is still a random element to their decision
                if random.random() < p:
                    decision = trunc(round(I),max_I,1)
                else:
                    decision = trunc(-abs(round(I)),-1,-max_I)

                # finally we record the decision and write it to the next timeline
                G[t+1,i] = decision
                D[investor_type[i], t,i] = decision

            # trader influences non-active neighbour to join
            if G[t,i] != 0:
                stance = G[t,i]
                if random.random() < ph:
                    if G[t,(i-1)%N1] == 0 and G[t,(i+1)%N1] == 0:
                        ni = np.random.choice(choice)
                        G[t+1,(i+ni)%N1] = np.random.choice(choice)
                    elif G[t,(i-1)%N1] == 0:
                        G[t+1,(i-1)%N1] = np.random.choice(choice)
                    elif G[t,(i+1)%N1] == 0:
                        G[t+1,(i+1)%N1] = np.random.choice(choice)
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
        sum_called_shares = np.sum(P[t] * margin_call)
        sum_margin_called = np.sum(margin_call)

        # these shares are sold at current price
        U[t+1] = sum_called_shares
        stack += sum_called_shares * sum_margin_called


    final_trade = P[-1] * S[-1]
    B[-1] += final_trade
    N[-1] = B[-1]

    return (G,P,N,S,X,D,T,U,C, initial_account_balance)

# %%

if __name__ == "__main__":

    G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(
        trigger = True, bound = True, pd = 0.05, pe = 0.01,
        ph = 0.0485, pa = 0.7, N0=2000, N1 =2000, A =4, a=1, h=1, 
        pi1 = 0.5, pi2 = 0.3, pi3 = 0.2, ub=1000, lb=20)
    
    plt.figure(figsize=(8,4))
    plt.imshow(G.T, interpolation="None", aspect="auto", cmap="bwr")

# %%
