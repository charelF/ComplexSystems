#%%%###################################################################
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import pandas as pd


import sys
sys.path.append('../IDL')


import entropy_estimators as ee

#%%%###################################################################
def MI(s1, s2):
    l = len(s1)
    v1 = s1.reshape(l, 1)
    v2 = s2.reshape(l, 1)
    return ee.mi(v1, v2)

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

res = euler(M=365, T=1, S=100, sigma=0.2, r=0.06, num_series=50, rho=0.3) 

for i in range(0, res.shape[0]):
    plt.plot(res[i, :])

res_log_return = np.zeros((res.shape[0],res.shape[1]-1))
for i in range(0, res.shape[1]-1):
    for j in range(0,res.shape[0]):
        res_log_return[j,i] = np.log(res[j,i]) - np.log(res[j,i+1])

plt.plot(res_log_return)

plt.matshow(res_log_return)
df = pd.DataFrame(res_log_return.T)
plt.matshow(df.corr(method=MI), vmax=1)

#%%%###################################################################

rho_range = np.linspace(0,3,20)
sigma_range = np.linspace(0,0.1,50)


system_MI = np.zeros((sigma_range.shape[0], rho_range.shape[0]))
for x, sigma_val in enumerate(sigma_range):
    for k, rho_val in enumerate(rho_range):
        res = euler(M=365, T=1, S=20000, sigma=sigma_val, r=0, num_series=10, rho=rho_val) 

        res_log_return = np.zeros((res.shape[0],res.shape[1]-1))
        for i in range(0, res.shape[1]-1):
            for j in range(0,res.shape[0]):
                if res[j,i]<=0 or res[j,i+1]<=0:
                    res_log_return[j,i] = np.nan()
                else:
                    res_log_return[j,i] = np.log(res[j,i]) - np.log(res[j,i+1])

        df = pd.DataFrame(res_log_return.T)
        MI_matrix = df.corr(method=MI)
        system_MI[x, k] =np.mean(np.array(MI_matrix).flatten())
        


#%%%###################################################################
plt.figure(figsize=(15,5))
for i in range(sigma_range.shape[0]):
    plt.plot(rho_range, system_MI[i,:]/np.max(system_MI), label = f'{sigma_range[i]}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

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