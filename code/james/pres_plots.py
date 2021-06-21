# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
from scipy import stats
from numba import jit

import sys
sys.path.append('../shared')
from wednesdaySPEED import simulation
from analytic_tools import gen_hurst_exponent

# %%

res = np.zeros((20, 30))
for z in range(20):
    G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = True, pd = 0.05, pe = 0.01,
        ph = 0.0485, pa = 0.7, N0=1000, N1 = 100, A = 4, a=1, h=1, 
        pi1 = 0.5, pi2 = 0.3, pi3 = 0.2)
    h_res, q_vals = gen_hurst_exponent(S, 30)
    res[z,:] = h_res*q_vals

res_mean_ca = np.mean(res, axis=0)
res_std_ca = np.std(res, axis=0)

#%%

df = pd.read_csv("../../data/all_world_indices_clean.csv")
print(df.columns)
df_spx = df[["Date", "SPX Index"]]
df_spx["Date"] = pd.to_datetime(df_spx["Date"], format='%d/%m/%Y')
df_spx = df_spx.sort_values(by="Date")
df_spx.reset_index(inplace=True)
series_array = np.array(df_spx["SPX Index"])

## identical to np.split but doesnt raise exception if arrays not equal length
split = np.array_split(series_array, 6)
res = np.zeros((6, 30))

for i in range(len(split)):
    h_res, q_vals = gen_hurst_exponent(split[i], 30)
    res[i,:] = h_res*q_vals

res_mean_sp = np.mean(res, axis=0)
res_std_sp = np.std(res, axis=0)

#%%

df = pd.read_csv("../../data/all_world_indices_clean.csv")
df_nky = df[["Date", "NKY Index"]]
df_nky["Date"] = pd.to_datetime(df_nky["Date"], format='%d/%m/%Y')
df_nky = df_nky.sort_values(by="Date")
df_nky.reset_index(inplace=True)
series_array = np.array(df_nky["NKY Index"])

## identical to np.split but doesnt raise exception if arrays not equal length
split = np.array_split(series_array, 6)
res = np.zeros((6, 30))

for i in range(len(split)):
    h_res, q_vals = gen_hurst_exponent(split[i], 30)
    res[i,:] = h_res*q_vals

res_mean_nky = np.mean(res, axis=0)
res_std_nky = np.std(res, axis=0)

#%%

fig, (ax1, ax2, ax3) = plt.subplots(
    ncols=1, nrows=3, figsize=(12,8), sharex=True, gridspec_kw = {'wspace':0, 'hspace':0}
)

ax1.errorbar(q_vals, res_mean_ca, color="C4", yerr=res_std_ca, label='Rule-Based CA')
ax1.fill_between(q_vals, res_mean_ca + res_std_ca, res_mean_ca - res_std_ca, color="C4", alpha=0.2)
ax1.grid(alpha=0.2)
ax1.set_ylabel(r"$q \cdot H(q)$")
ax1.set_xlabel(r"$q$")
ax1.legend()

ax2.errorbar(q_vals, res_mean_sp, color="C6", yerr=res_std_sp, label='S&P500')
ax2.fill_between(q_vals, res_mean_sp + res_std_sp, res_mean_sp - res_std_sp, color="C6", alpha=0.2)
ax2.grid(alpha=0.2)
ax2.set_ylabel(r"$q \cdot H(q)$")
ax2.set_xlabel(r"$q$")
ax2.legend()

ax3.errorbar(q_vals, res_mean_nky, color="C5", yerr=res_std_nky, label='IGBVL')
ax3.fill_between(q_vals, res_mean_nky + res_std_nky, res_mean_nky - res_std_nky, color="C5", alpha=0.2)
ax3.grid(alpha=0.2)
ax3.set_ylabel(r"$q \cdot H(q)$")
ax3.set_xlabel(r"$q$")
ax3.legend()

plt.savefig("imgs/img_hurst_exp_triple", dpi=300)
# %%

fig = plt.figure(figsize=(12, 8))

plt.plot(q_vals, res_mean_ca, 'v-', label='Rule-Based CA')
plt.fill_between(q_vals, res_mean_ca + res_std_ca, res_mean_ca - res_std_ca, alpha=0.1)
plt.plot(q_vals, res_mean_sp, 'x-', label='S&P500')
plt.fill_between(q_vals, res_mean_sp + res_std_sp, res_mean_sp - res_std_sp, alpha=0.1)
plt.plot(q_vals, res_mean_nky, 'o-', label='IGBVL')
plt.fill_between(q_vals, res_mean_nky + res_std_nky, res_mean_nky - res_std_nky, alpha=0.1)
plt.grid(alpha=0.2)
plt.ylabel(r"$q \cdot H(q)$")
plt.xlabel(r"$q$")
plt.legend()
plt.savefig("imgs/img_hurst_exp_triple", dpi=300)
plt.show()

# %%
