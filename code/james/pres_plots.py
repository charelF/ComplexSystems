# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
from scipy.optimize import curve_fit
import scipy.stats as stats
from numba import njit, prange
import statsmodels.api as sm
import math
import itertools
import operator

import sys
sys.path.append('../shared')
from wednesdaySPEED import simulation
from analytic_tools import gen_hurst_exponent, count_crashes
from original_implementation import execute

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

## new model
G_new, P, N, S, X, D, T, U, C, initial_account_balance = simulation(trigger = False, bound = True, pd = 0.05, pe = 0.01,
    ph = 0.0485, pa = 0.7, N0=1000, N1 = 4000, A = 4, a=1, h=1, 
    pi1 = 0.5, pi2 = 0.3, pi3 = 0.2)

## old model 
r, G_old = execute(0.1, 0.0001, 0.1, 2, 0.1, 0.1)

# %%

np.save("arrs/G_new_big_dist", G_new)
np.save("arrs/G_old_big_dist", G_old)

# %%

def power_law(x, a, b):
    return a * x ** (b)

clusters_old = [[i for i,value in it] for key,it in itertools.groupby(enumerate(G_old[-1,:]), 
                                                                  key=operator.itemgetter(1)) if key != 0]

clusters_new = [[i for i,value in it] for key,it in itertools.groupby(enumerate(G_new.T[-1,:]), 
                                                                  key=operator.itemgetter(1)) if key != 0]
cluster_size_new, cluster_size_old = [], []

for i in range(len(clusters_new)):
    cluster_size_new.append(len(clusters_new[i]))

for i in range(len(clusters_old)):
    cluster_size_old.append(len(clusters_old[i]))

unique_old, counts_old = np.unique(cluster_size_old, return_counts=True)
unique_new, counts_new = np.unique(cluster_size_new, return_counts=True)

popt_old, _ = curve_fit(power_law, unique_old, counts_old)
popt_new, _ = curve_fit(power_law, unique_new, counts_new)


# %%

print(unique_old, counts_old)
print(unique_new, counts_new)

fig, ax = plt.subplots() 
ax.scatter(unique_old, counts_old, color="C0", marker="v")
ax.scatter(unique_new, counts_new, color="C1")

power_law_old = popt_old[0]*unique_old**(popt_old[1])
power_law_new = popt_new[0]*unique_new**(popt_new[1])

ax.plot(unique_old, power_law_old, color='C0', label=f'Bartolozzi Automata', ls='--')
ax.plot(unique_new, power_law_new, color='C1', label=f'Quasi-deterministic Automata', ls='--')

# ax.plot(unique_old, power_law_old, color='C4', label=f'Bartolozzi lambda~{-1 * popt_old[1]:.2f}', ls='--')
# ax.plot(unique_new, power_law_new, color='C7', label=f'lambda~{-1 * popt_new[1]:.2f}', ls='--')

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('S')
ax.set_ylabel(r'$\rho$')
ax.grid(alpha=0.2, which="major")
ax.grid(alpha=0.05, which="minor")
ax.legend()
plt.savefig("imgs/power_law_old_vs_new")
plt.show()

# %%

## return distribution 

df = pd.read_csv("../../data/all_world_indices_clean.csv")
df_spx = df[["Date", "SPX Index"]]
df_spx["Date"] = pd.to_datetime(df_spx["Date"], format='%d/%m/%Y')
df_spx = df_spx.sort_values(by="Date")
df_spx.reset_index(inplace=True)
series_array = np.array(df_spx["SPX Index"])
log_ret_dat = np.diff(np.log(series_array))
log_ret_dat_stan = (log_ret_dat - np.mean(log_ret_dat)) / np.std(log_ret_dat)

G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = True, pd = 0.05, pe = 0.01,
        ph = 0.0485, pa = 0.7, N0=4000, N1 = 100, A = 4, a=1, h=1, 
        pi1 = 0.3, pi2 = 0.5, pi3 = 0.2)
r = (X - np.mean(X)) / np.std(X)

#%%

plt.hist(r, alpha=0.4, bins=30, label="Automaton", density=True)
plt.hist(log_ret_dat_stan, bins=30, alpha=0.4, label="S&P 500", density=True)
plt.yscale("log")
##plt.title("Log Return Distribution - Standardised")
plt.legend()
plt.grid(alpha=0.2)
plt.ylabel(r"$\rho$")
plt.xlabel(r"$r$")
plt.savefig("imgs/standardised_dist_plot", dpi=300)
plt.show()
    
## 2 independent samples KS test
ksstat, pval = stats.ks_2samp(r, log_ret_dat_stan)
print(ksstat, pval)

# %%

N_range = [10, 50, 250]
A_range = np.linspace(0.2, 10, 50)
sims = 150

# NA_res = np.zeros((sims, len(N_range), A_range.shape[0]))

# for j, N_val in enumerate(N_range):
#     for i, A_val in enumerate(A_range):

#         for k in range(sims):
#             print(N_val, A_val, k)
#             G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = False, pd = 0.05, pe = 0.01,
#                     ph = 0.0485, pa = 0.7, N0=1000, N1 = N_val, A = A_val, a=1, h=1, 
#                     pi1 = 0.8, pi2 = 0.2, pi3 = 0)

#             NA_res[k, j, i] = count_crashes(X, 0.65, window=5) 

# %%

np.save("arrs/crashes_A_N", NA_res)
# %%

NA_res = np.load("crashes_A_N.npy")
mn = np.mean(NA_res, axis=0)
sd = 1.96 * np.std(NA_res, axis=0) / np.sqrt(sims)

for z in range(mn.shape[0]):
    plt.plot(A_range, mn[z,:], label=f"Traders = {N_range[z]}")
    plt.fill_between(A_range, mn[z,:] - sd[z,:], mn[z,:] + sd[z,:], alpha = 0.2)
plt.legend(loc=2)
plt.ylabel("Number of Crashes")
plt.xlabel("A")
plt.grid(alpha=0.2)
plt.savefig("imgs/crash_simulation_A_N_2", dpi = 300)
plt.xlim(2.5, 10)
plt.show()


# %%

df = pd.read_csv("../../data/all_world_indices_clean.csv")
df_spx = df[["Date", "SPX Index"]]
df_spx["Date"] = pd.to_datetime(df_spx["Date"], format='%d/%m/%Y')
df_spx = df_spx.sort_values(by="Date")
df_spx.reset_index(inplace=True)
series_array = np.array(df_spx["SPX Index"])
log_ret_dat = np.diff(np.log(series_array))
log_ret_dat_stan = (log_ret_dat - np.mean(log_ret_dat)) / np.std(log_ret_dat)

G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = True, pd = 0.05, pe = 0.01,
        ph = 0.0485, pa = 0.7, N0=4000, N1 = 100, A = 4, a=1, h=1, 
        pi1 = 0.3, pi2 = 0.5, pi3 = 0.2)
r = (X - np.mean(X)) / np.std(X)

# %%

## x and y data
acf_x = sm.tsa.stattools.acf(r)[1:]
acf_sp = sm.tsa.stattools.acf(log_ret_dat_stan[1:])[1:]

acf_x_vol = sm.tsa.stattools.acf(np.abs(r))[1:]
acf_sp_vol = sm.tsa.stattools.acf(np.abs(log_ret_dat_stan[1:]))[1:]

x = np.arange(acf_x.shape[0])

## mean above which the acf is significantly different from 
## zero at the 95% level. 2.021 is the critical value 
threshold = 2.021 * np.std(acf_x)

fig = plt.figure(figsize=(15, 5))
plt.plot(x, acf_x, label="S&P500 Returns")
plt.plot(x, acf_sp, label="Automaton Returns")
plt.plot(x, [threshold]*acf_x.shape[0], "--", color="C2", alpha=0.4)
plt.plot(x, [-1 * threshold]*acf_x.shape[0], "--", color="C2", alpha=0.4)
plt.xlabel("Lag")
plt.ylabel("Autocorrelations")
plt.grid(alpha=0.2)
plt.legend()
plt.savefig("imgs/acf_ret")
plt.show()
# %%

fig, (ax1, ax2) = plt.subplots(
        ncols=1, nrows=2, figsize=(12,7), sharex=True, gridspec_kw = {'wspace':0, 'hspace':0.15})
        
ax1.plot(x, acf_x, label="S&P500 Returns")
ax1.plot(x, acf_sp, label="Automaton Returns")
ax1.plot(x, [threshold]*acf_x.shape[0], "--", color="black", alpha=0.5, label="Significance Threshold")
ax1.plot(x, [-1 * threshold]*acf_x.shape[0], "--", color="black", alpha=0.5)
ax1.fill_between(x, [threshold]*acf_x.shape[0], [-1 * threshold]*acf_x.shape[0], color="black", alpha=0.05) 
ax1.grid(alpha=0.2)
ax1.set_ylabel(r"Autocorrelation")
ax1.legend() 

ax2.plot(x, acf_x_vol, color="C0", label="S&P500 Volatility")
ax2.plot(x, acf_sp_vol, color="C1", label="Automaton Volatility")
ax2.plot(x, [threshold]*acf_x.shape[0], "--", color="black", alpha=0.5, label="Significance Threshold")
ax2.fill_between(x, [threshold]*acf_x.shape[0], [threshold - 0.04]*acf_x.shape[0], color="black", alpha=0.05) 
##ax2.plot(x, [-1 * threshold]*acf_x.shape[0], "--", color="C2", alpha=0.4)
ax2.grid(alpha=0.2)
ax2.set_ylabel(r"Autocorrelation")
ax2.legend() 

fig.align_ylabels()
plt.xlabel("Lag")
plt.savefig("imgs/acf_double_plot", dpi=300)
##plt.xlim(200, 500)
plt.show()

# %%

N_range = [10, 50, 250]
pi1_range = np.linspace(0.9, 0, 30)
sims = 20

Npi_res = np.zeros((sims, len(N_range), pi1_range.shape[0]))

for j, N_val in enumerate(N_range):
    for i, pi1_val in enumerate(pi1_range):

        for k in range(sims):
            print(N_val, pi1_val, k)
            G,P,N,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = True, pd = 0.05, pe = 0.01,
                    ph = 0.0485, pa = 0.7, N0=1500, N1 = N_val, A = 4.5, a=1, h=1, 
                    pi1 = pi1_val, pi2 = 0.9 - pi1_val, pi3 = 0.1)

            Npi_res[k, j, i] = count_crashes(X, 0.9, window=5) 

# %%

np.save("arrs/crashes_pi1_N", Npi_res)
# %%

##NA_res = np.load("crashes_A_N.npy")
mn = np.mean(Npi_res, axis=0)
sd = 1.96 * np.std(Npi_res, axis=0) / np.sqrt(sims)

print(mn)
print(sd)

for z in range(mn.shape[0]):
    plt.plot(pi1_range, mn[z,:], label=f"Traders = {N_range[z]}")
    plt.fill_between(pi1_range, mn[z,:] - sd[z,:], mn[z,:] + sd[z,:], alpha = 0.2)
plt.legend(loc=2)
plt.ylabel("Number of Crashes")
plt.xlabel("Stochastic Trader Ratio")
plt.grid(alpha=0.2)
plt.xlim(0.4, 0.9)
plt.savefig("imgs/crash_simulation_pi1_N", dpi = 300)
plt.show()

# %%
