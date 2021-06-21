# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import sys
sys.path.append("../shared")
from analytic_tools import fractal_latent_heat_alex
from wednesdaySPEED import simulation

# %%
tau = 9
pi_2_vals = [0.0, 0.1, 0.2, 0.3, 0.5]

plt.figure(figsize=(10,5))
for i, val in enumerate(pi_2_vals):
    G,P,_,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = False, pd = 0.05, pe = 0.01,
                    ph = 0.0485, pa = 0.7, N0=1000, N1 = 100, A = 4, a=1, h=1, 
                    pi1 = 0.5, pi2 = val, pi3 = 0.5 - val)

    q_vals, C_q, S_q, X_q = fractal_latent_heat_alex(np.array(S), tau, 100)
    plt.plot(q_vals[2:], -S_q[:-1], label=f"Pi2 = {val}")
    
plt.ylabel("S - Entropy")
plt.xlabel("Temperature")
plt.xlim(-20, 20)
plt.grid()
plt.legend()
plt.show()

# %%

tau = 9
pi_2_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

plt.figure(figsize=(10,5))
for i, val in enumerate(pi_2_vals):
    G,P,_,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = False, pd = 0.05, pe = 0.01,
                    ph = 0.0485, pa = 0.7, N0=1000, N1 = 100, A = 4, a=1, h=1, 
                    pi1 = 0.5, pi2 = val, pi3 = 0.5 - val)

    q_vals, C_q, S_q, X_q = fractal_latent_heat_alex(np.array(S), tau, 100)
    plt.plot(q_vals[2:],C_q, label=f"Pi2 = {val}")

plt.ylabel("C_p - Specific heat")
plt.xlabel("Temperature")
plt.xlim(-5, 5)
plt.grid()
plt.legend()
plt.show()

# %%

tau = 9
pi_2_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

plt.figure(figsize=(10,5))
for i, val in enumerate(pi_2_vals):
    G,P,_,S,X,D,T,U,C, initial_account_balance = simulation(trigger = False, bound = False, pd = 0.05, pe = 0.01,
                    ph = 0.0485, pa = 0.7, N0=1000, N1 = 100, A = 4, a=1, h=1, 
                    pi1 = 1 - val, pi2 = val, pi3 = 0)

    q_vals, C_q, S_q, X_q = fractal_latent_heat_alex(np.array(S), tau, 100)
    plt.plot(q_vals[2:], -S_q[:-1], label=f"Pi2 = {val}")
    
plt.ylabel("S - Entropy")
plt.xlabel("Temperature")
plt.xlim(-20, 20)
plt.grid()
plt.legend()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plt.figure(figsize=(10,5))
# plt.plot(q_vals/40, X_q/np.max(X_q), c="r", label="Free Energy - H")
# plt.plot(q_vals[2:]/40, -S_q[:-1]/np.max(-S_q), c="b", label="Entropy - dH/dT")
# plt.plot(q_vals[2:]/40,C_q/np.max(C_q), c="g", label="Specific heat- dH^2/dT^2")
# plt.ylabel("")
# plt.xlabel("Temperature")
# plt.legend()
# plt.show()


# plt.figure(figsize=(10,5))
# plt.plot(q_vals, X_q)
# plt.ylabel("H - Free Energy")
# plt.xlabel("Temperature")
# plt.show()

plt.figure(figsize=(10,5))
plt.plot(q_vals[2:], -S_q[:-1])
plt.ylabel("S - Entropy")
plt.xlabel("Temperature")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(q_vals[2:],C_q)
plt.ylabel("C_p - Specific heat")
plt.xlabel("Temperature")
plt.show()

# %%

# %%
