#%%%###################################################################
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import pandas as pd
import scipy
import sys
sys.path.append('../IDL')


import entropy_estimators as ee

#%%%###################################################################
def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

def MI(s1, s2):
    l = len(s1)
    v1 = s1.reshape(l, 1)
    v2 = s2.reshape(l, 1)
    return ee.mi(v1, v2)


def compute_IDL(df, plot=True):
    l, w = df.shape
    IDL_list = []
    
    for i,idx1 in enumerate(df):
        mutual_info_list = []
        for j,idx2 in enumerate(df):
            if idx1 == idx2: continue
            s1 = df[idx1].to_numpy().reshape(l, 1)
            s2 = df[idx2].to_numpy().reshape(l, 1)
            mutual_info_list.append(ee.mi(s1, s2))

        y = sorted(mutual_info_list)[::-1]
        x = np.arange(len(y))

        slope = (np.max(y) - np.min(y)) / len(y)
        intercept = np.max(y)
        sol = np.max(y)/(2*slope)

        try:
            popt, pcov = scipy.optimize.curve_fit(exp_decay, x, y, maxfev=2500)
            hl = ((np.log(0.5)-popt[2])/-popt[1])
            if abs(hl) < w:
                IDL = abs(hl)
            else:
                raise ValueError("fitted half life was larger than total number of indices")
        except Exception as e:
            if plot: print(f"[ERROR] {e}")
            IDL = min(abs(sol), w)  # if even sol fails we use max length
            # This is potentially problematic

        IDL_list.append(IDL)

        # visualisation
        if plot:
            plt.title(idx1)
            plt.plot(x, y, ".-", alpha=0.5, label="mi sorted")
            if (IDL == sol) or (IDL == w):
                ymin, ymax = plt.gca().get_ylim()
                plt.plot(x, x*(-slope)+intercept, "--", label="linear fit", color="C2")
                plt.vlines(x=IDL, ymax=ymax, ymin=ymin, linestyle="--", color="blue", label="linear IDL")
            else:
                ymin, ymax = plt.gca().get_ylim()
                plt.plot(x, exp_decay(x, *popt), "-", label="exp decay fit", color="C1")
                plt.vlines(x=IDL, ymax=ymax, ymin=ymin, linestyle="--", color="red", label="exp decay IDL")
            plt.legend()
            plt.show()
            
    return pd.Series(index=df.columns, data=IDL_list)

#%%%###################################################################
def euler(M, T, S, sigma, r, num_series, rho):
    dt = T/M
    S_values = np.zeros((num_series, M))
    S_values[:,0] = S

    for m in range(1, M):

        shared_Zm = random.gauss(0, 1)
        for n in range(0, num_series):
            Zm = random.gauss(0, 1)
            S_values[n, m] = S_values[n, m-1] + r*S_values[n, m-1]*dt + sigma*S_values[n, m-1]*math.sqrt(dt)*Zm + rho*S_values[n, m-1]*math.sqrt(dt)*shared_Zm 


    return S_values
#%%%###################################################################

res = euler(M=1000, T=1, S=100, sigma=0.2, r=0.06, num_series=50, rho=0.1) 

for i in range(0, res.shape[0]):
    plt.plot(res[i, :])

res_log_return = np.zeros((res.shape[0],res.shape[1]-1))
for i in range(0, res.shape[1]-1):
    for j in range(0,res.shape[0]):
        res_log_return[j,i] = np.log(res[j,i]) - np.log(res[j,i+1])

plt.plot(res_log_return)

plt.matshow(res_log_return)
df = pd.DataFrame(res_log_return.T)
plt.matshow(df.corr(method=MI),vmax=1 )



#%%%###################################################################


def euler(M, T, S, sigma, sigma2_val, r, num_series, rho, probability_threshold):
    dt = T/M
    S_values = np.zeros((num_series, M))
    S_values[:,0] = np.random.uniform(low=3*S, high=6*S, size = num_series)

    MARKET = np.zeros((M))
    MARKET[0] = np.mean(S_values[:,0])

    p = np.random.uniform(low=0, high=1, size=num_series)
    corr = np.random.exponential(scale=1, size = num_series)
1
    for m in range(1, M):
        sigma2 =random.gauss(sigma2_val, 0.5)
        for n in range(0, num_series):
            Zm = random.gauss(0, 1)
            r_val = random.gauss(r, 0.3)
            sigma_val =random.gauss(sigma, 0.5*sigma)
            

            if p[n]>=probability_threshold:
                S_values[n, m] = S_values[n, m-1] + r_val*S_values[n, m-1]*dt + sigma_val*S_values[n, m-1]*math.sqrt(dt)*Zm + rho*corr[n]/np.max(corr)* sigma2*S_values[n, m-1]*math.sqrt(dt) 
            else:
                S_values[n, m] = S_values[n, m-1] + r_val*S_values[n, m-1]*dt + sigma_val*S_values[n, m-1]*math.sqrt(dt)*Zm 

            if S_values[n, m]==0:
                S_values[n, m]=10
        
        MARKET[m] = np.mean(S_values[:,m])
        if MARKET[m]-MARKET[m-1]< 0:
            rho = -1
        elif MARKET[m]-MARKET[m-1]> 0:
            rho = +1
        else:
            rho=0

    return S_values, MARKET
#%%%###################################################################
plt.figure(figsize=(15,5))

stocks, market = euler(M=2000, T=1, S=1000, sigma=0.2,sigma2_val = 0.4, r=0.06, 
num_series=20, rho=1, probability_threshold=0) 

for i in range(0, 10):
    plt.plot(stocks[i, :], label= f'STOCK_{i}', color="black")
plt.plot(market, label="MARKET", color="red")
plt.legend()
#%%%###################################################################
log_return = np.zeros((stocks.shape[0],stocks.shape[1]-1))

for i in range(0, stocks.shape[1]-1):
    for j in range(0, stocks.shape[0]):
        log_return[j,i] = np.log(stocks[j,i+1]) - np.log(stocks[j,i])

df = pd.DataFrame(log_return.T)
plt.matshow(df.corr(), vmin=0, vmax=1)
plt.colorbar()
#%%%###################################################################
IDL = np.mean(compute_IDL(df, plot=False))
print(IDL)
#%%%###################################################################
probability_threshold_range = np.linspace(0,1,10)
sigma_range = np.linspace(0,1,10)


system_MI = np.zeros((sigma_range.shape[0], probability_threshold_range.shape[0]))

for x, sigma_val in enumerate(sigma_range):
    for k, probability_threshold_val in enumerate(probability_threshold_range):
        res, market = euler(M=2000, T=1, S=1000, sigma=sigma_val, sigma2_val=0.5, r=0.06, 
        num_series=10, rho=1, probability_threshold=probability_threshold_val) 

        res_log_return = np.zeros((res.shape[0],res.shape[1]-1))
        for i in range(0, res.shape[1]-1):
            for j in range(0,res.shape[0]):

                    res_log_return[j,i] = np.log(res[j,i+1]) - np.log(res[j,i])
        
        df = pd.DataFrame(res_log_return.T)
        MI_matrix = df.corr(method=MI)
        system_MI[x, k] =np.mean(np.array(MI_matrix).flatten())
        print(np.mean(compute_IDL(df, plot=False)))
        system_IDL[x, k] = np.mean(compute_IDL(df, plot=False))
#%%%###################################################################
plt.figure(figsize=(15,5))
for i in range(int(probability_threshold_range.shape[0])):
    plt.plot(sigma_range, system_IDL[:,i], label = f'{probability_threshold_range[i]}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel("IDL")
    plt.xlabel("Sigma_value")


#%%%###################################################################
plt.figure(figsize=(15,5))
for i in range(int(probability_threshold_range.shape[0])):
    plt.plot(sigma_range, system_MI[:,i], label = f'{probability_threshold_range[i]}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel("MI")
    plt.xlabel("Sigma_value")






#%%%###################################################################
IDL_mean = np.mean(compute_IDL(df, plot=True))


#%%%###################################################################

rho_range = np.linspace(0,1,100)
sigma_range = np.linspace(0,1,10)


system_MI = np.zeros((sigma_range.shape[0], rho_range.shape[0]))
for x, sigma_val in enumerate(sigma_range):
    for k, rho_val in enumerate(rho_range):
        res = euler(M=300, T=1, S=20000, sigma=sigma_val, r=0, num_series=10, rho=rho_val) 

        res_log_return = np.zeros((res.shape[0],res.shape[1]-1))
        for i in range(0, res.shape[1]-1):
            for j in range(0,res.shape[0]):
                if res[j,i]<=0 or res[j,i+1]<=0:
                    res_log_return[j, i] = np.nan
                else:
                    res_log_return[j,i] = np.log(res[j,i]) - np.log(res[j,i+1])

        df = pd.DataFrame(res_log_return.T)
        # MI_matrix = df.corr(method=MI)
        # system_MI[x, k] =np.mean(np.array(MI_matrix).flatten())
        system_MI[x, k] = np.mean(compute_IDL(df, plot=False))


#%%%###################################################################
plt.figure(figsize=(15,5))
for i in range(int(sigma_range.shape[0]*0.5)):
    plt.plot(rho_range/sigma_range, system_MI[i,:], label = f'{sigma_range[i]}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel("IDL")
    plt.xlabel("RHO")
#%%%###################################################################
plt.figure(figsize=(15,5))
for i in range(sigma_range.shape[0]):
    plt.plot(rho_range, np.log(system_MI[i,:]/np.max(system_MI)), label = f'{sigma_range[i]}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# %%
plt.figure(figsize=(15,5))
for i in range(sigma_range.shape[0]):
    plt.plot(rho_range, system_MI[i,:]/np.max(system_MI), label = f'{sigma_range[i]}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')