import numpy as np 
import pandas as pd
from numba import njit, prange, jit

def gen_hurst_exponent(time_series, num_q, max_lag=200):
    """
    Returns the Hurst Exponent of the time series taised to the power 
    - time_series: the series of to find exponent values for 
    - num_q: the density of q (equivelent temperature)
    - max_lag: the maximum value of tau to fit for 
        - default = 200
    """
    
    lags = range(2, max_lag)
    q_vals = np.linspace(1, 5, num_q) 
    S_q = np.zeros(len(lags))
    reg = np.zeros(len(q_vals))
    for i, q_val in enumerate(q_vals):
        for j, lag in enumerate(lags):
            S_q[j] = np.mean(np.abs(time_series[lag:]-time_series[:-lag])**q_val)
         # calculate the slope of the log plot -> the Hurst Exponent
        reg[i] = np.polyfit(np.log(lags), np.log(S_q), 1)[0]

    return reg/q_vals, q_vals

def fractal_latent_heat(series, tau, N):
    '''
    fractal latent heat values for a given time series
    - series: the vector for which to calculate latent heat values
    - tau: the time offset to use for calculating normalised probability
    - N: the number of chunks into which we will split the array 
        - this is equivelent the the number of normalised probability measures we will end up with
    '''

    ## how many chunks
    splt = np.array_split(series, N)
    q_vals = np.linspace(-6, 6, 30)

    ## structs
    C_q = np.zeros(q_vals.shape[0] - 2) 
    S_q = np.zeros(q_vals.shape[0] - 1) 
    X_q = np.zeros(q_vals.shape[0])
    mu_i = np.zeros(len(splt))
    denom_sum = 0

    ## eq 10
    for i in range(len(splt)):
        denom_sum += np.abs(splt[i][tau] - splt[i][0])

    for j in range(len(splt)):
        mu_i[j] = np.abs(splt[j][tau] - splt[j][0]) / denom_sum

    lhs = np.zeros((q_vals.shape[0]))
    rhs = np.zeros((q_vals.shape[0]))

    for k, val in enumerate(q_vals):
        ## eq 11
        lhs[k] = np.log(np.sum(mu_i**val))
        rhs[k] = np.log(N)
        ## solve for slope of log-log
        ## x_q equivelent to tau(q) in casenna
        X_q[k] = lhs[k] / rhs[k]

    for l in range(1, q_vals.shape[0] - 1):
        C_q[l - 1] = X_q[l + 1] - 2 * X_q[l] + X_q[l - 1]
        S_q[l - 1] = X_q[l + 1] - X_q[l - 1]

    return q_vals, C_q, S_q

def fractal_latent_heat_alex(series, tau, N):
    '''
    this is equivelent to the function above except that now 
    the values for the scaling exponent X_q are also returned
    - series: the vector for which to calculate latent heat values
    - tau: the time offset to use for calculating normalised probability
    - N: the number of chunks into which we will split the array 
    '''
    splt = np.array_split(series, N)
    q_vals = np.linspace(-50, 50, 1000)

    ## structs
    C_q = np.zeros(q_vals.shape[0] - 2) 
    X_q = np.zeros(q_vals.shape[0])
    S_q = np.zeros(q_vals.shape[0] - 1)
    mu_i = np.zeros(len(splt))
    denom_sum = 0

    ## eq 10
    for i in range(len(splt)):
        denom_sum += np.abs(splt[i][tau] - splt[i][0])

    for j in range(len(splt)):
        mu_i[j] = np.abs(splt[j][tau] - splt[j][0]) / denom_sum

    lhs = np.zeros((q_vals.shape[0]))
    rhs = np.zeros((q_vals.shape[0]))

    for k, val in enumerate(q_vals):
        ## eq 11
        lhs[k] = np.log(np.sum(mu_i**val))
        rhs[k] = np.log(N)
        ## solve for slope of log-log
        ## x_q equivelent to tau(q) in casenna
        X_q[k] = lhs[k] / rhs[k]

    # ## cannot obtain C_q for first and last q vals
    for l in range(1, q_vals.shape[0] - 1):
        C_q[l - 1] = X_q[l + 1] - 2 * X_q[l] + X_q[l - 1]
        S_q[l - 1] = X_q[l + 1] - X_q[l - 1]

    return q_vals, C_q, S_q, X_q

@jit(nopython=True)
def count_crashes(X, treshold, window=5):
    """
    does it better than james
    - X: log returns array, in range -1, 1
    - treshold: the log return that defines a crash: 
        - e.g. if 20% drop over 5 days = crash then the treshold should be 0.8
    - window: how many days: default: 5 days
    """

    crashes = 0
    for i in range(len(X)-window):
        period = X[i:i+window]+1
        prod = np.prod(period)
        geo_mean = prod ** (1/window)
        if geo_mean < treshold:
            crashes += 1

    return crashes


def get_crash_indices(X, treshold, window=5):
    """
    does it better than james
    - X: log returns array, in range -1, 1
    - treshold: the log return that defines a crash: 
        - e.g. if 20% drop over 5 days = crash then the treshold should be 0.8
    - window: how many days: default: 5 days
    """

    crashes = []
    for i in range(len(X)-window):
        period = X[i:i+window]+1
        prod = np.prod(period)
        geo_mean = prod ** (1/window)
        if geo_mean < treshold:
            crashes.append(i)

    return crashes