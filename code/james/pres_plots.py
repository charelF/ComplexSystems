# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
from scipy import stats
import statsmodels.api as sm

from numba import jit

import sys
sys.path.append('../shared')
from wednesdaySPEED import simulation

# %%

res = np.zeros((20, 30))
for z in range(20):
    G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = True, pd = 0.05, pe = 0.01,
        ph = 0.0485, pa = 0.7, N0=1000, N1 = 100, A = 4, a=1, h=1, 
        pi1 = 0.5, pi2 = 0.3, pi3 = 0.2)
    h_res, q_vals = get_hurst_exponent(S)
    res[z,:] = h_res*q_vals

res_mean_ca = np.mean(res, axis=0)
res_std_ca = np.std(res, axis=0)

#%%

df = pd.read_csv("../../data/all_world_indices_clean.csv")
df_spx = df[["Date", "SPX Index"]]
df_spx["Date"] = pd.to_datetime(df_spx["Date"], format='%d/%m/%Y')
df_spx = df_spx.sort_values(by="Date")
df_spx.reset_index(inplace=True)
series_array = np.array(df_spx["SPX Index"])

## identical to np.split but doesnt raise exception if arrays not equal length
split = np.array_split(series_array, 6)
res = np.zeros((6, 30))

for i in range(len(split)):
    h_res, q_vals = get_hurst_exponent(split[i])
    res[i,:] = h_res*q_vals

res_mean_sp = np.mean(res, axis=0)
res_std_sp = np.std(res, axis=0)

#%%

fig, (ax1,ax2) = plt.subplots(
    ncols=1, nrows=2, figsize=(12,8), sharex=True, gridspec_kw = {'wspace':0, 'hspace':0}
)

ax1.errorbar(q_vals, res_mean_ca, color="C4", yerr=res_std_ca, label='CA Gen')
ax1.fill_between(q_vals, res_mean_ca + res_std_ca, res_mean_ca - res_std_ca)
ax1.grid(alpha=0.2)
ax1.set_ylabel(r"$q \cdot H(q)$")
ax1.set_xlabel(r"$q$")
ax1.legend()

ax2.errorbar(q_vals, res_mean_sp, color="C6", yerr=res_std_sp, label='S&P500 Chunked')
ax1.fill_between(q_vals, res_mean_sp + res_std_sp, res_mean_sp - res_std_sp)
ax2.grid(alpha=0.2)
ax2.set_ylabel(r"$q \cdot H(q)$")
ax2.set_xlabel(r"$q$")
plt.legend()
plt.savefig("imgs/img_hurst_exp_double")

#%%

## return distribution 

df = pds.read_csv("../../data/all_world_indices_clean.csv")
df_spx = df[["Date", "SPX Index"]]
df_spx["Date"] = pds.to_datetime(df_spx["Date"], format='%d/%m/%Y')
df_spx = df_spx.sort_values(by="Date")
df_spx.reset_index(inplace=True)
series_array = np.array(df_spx["SPX Index"])
log_ret_dat = np.diff(np.log(series_array))
log_ret_dat_stan = (log_ret_dat - np.mean(log_ret_dat)) / np.std(log_ret_dat)

G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = True, pd = 0.05, pe = 0.01,
        ph = 0.0485, pa = 0.7, N0=1000, N1 = 100, A = 4, a=1, h=1, 
        pi1 = 0.5, pi2 = 0.3, pi3 = 0.2)
r = (X - np.mean(X)) / np.std(X)

#%%

fig = plt.figure(figsize=(8, 8))
plt.hist(r, alpha=0.4, bins=30, label="CA", density=True)
plt.hist(log_ret_dat_stan, bins=30, alpha=0.4, label="S&P500", density=True)
plt.yscale("log")
plt.title("Log Return Distribution - Standardised")
plt.legend()
plt.grid(alpha=0.2)
plt.show()

## 2 independent samples KS test
ksstat, pval = stats.ks_2samp(r, log_ret_dat_stan)
print(ksstat, pval)

# %%
