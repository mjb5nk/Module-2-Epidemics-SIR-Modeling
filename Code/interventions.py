'''
Notes from slides
• Each intervention will change some parameter (or move a population to a
new compartment) at day 70
• You will compare these changes to the baseline prediction for VT
• As you model them from days 70-120, compare the following to no
intervention:
• Peak infections with intervention
• Total cases (over days 70-120) prevented with intervention
• Cost & feasibility
• Compliance risk?
• Which would you recommend to VT?
'''

# 1. masking mandates
# Immediate masking mandate implemented from day 70 on, Reduces transmission by 40%


# 2. and 3. vaccine interventions
# 2. Vaccine campaign: single event on day 70, Vaccinate 2000 students on day 70 with 90% efficacy
# 3. Vaccine rollout: Vaccinate 1000 students on each: day 70, day 80, day 90


# 4. and 5. testing and quarantine interventions
# 4. Testing + quarantine starting day 70, Reduces infectious period by 2 days (due to delays in testing & low compliance)
# 5. Close school for 2 weeks, During closure, only 20% of normal contacts After closure, contact rate returns to normal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SEIR machinery (copied from Euler_Method.py)
def seir_model(y, t, beta, sigma, gamma, N):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

def euler_method(beta, sigma, gamma,
                 S0, E0, I0, R0, timepoints, N):
    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))
    S[0], E[0], I[0], R[0] = S0, E0, I0, R0
    for i in range(1, len(timepoints)):
        dt = timepoints[i] - timepoints[i-1]
        dSdt, dEdt, dIdt, dRdt = seir_model(
            (S[i-1], E[i-1], I[i-1], R[i-1]),
            timepoints[i-1], beta, sigma, gamma, N
        )
        S[i] = S[i-1] + dSdt * dt
        E[i] = E[i-1] + dEdt * dt
        I[i] = I[i-1] + dIdt * dt
        R[i] = R[i-1] + dRdt * dt
    return S, E, I, R

def grid_search_parameters(timepoints, N,
                           S0, E0, I0, R0,
                           observed_data,
                           beta_range=None,
                           sigma_range=None,
                           gamma_range=None):
    if beta_range is None:
        beta_range = np.linspace(0.1, 1.0, 10)
    if sigma_range is None:
        sigma_range = np.linspace(0.1, 1.0, 10)
    if gamma_range is None:
        gamma_range = np.linspace(0.1, 1.0, 10)

    sse_array = np.zeros((len(beta_range),
                          len(sigma_range),
                          len(gamma_range)))
    for b, beta in enumerate(beta_range):
        for s, sigma in enumerate(sigma_range):
            for g, gamma in enumerate(gamma_range):
                S, E, I, R = euler_method(
                    beta, sigma, gamma, S0, E0, I0, R0,
                    timepoints, N
                )
                sse_array[b, s, g] = np.sum((I - observed_data) ** 2)
    min_idx = np.unravel_index(np.argmin(sse_array), sse_array.shape)
    return (beta_range[min_idx[0]],
            sigma_range[min_idx[1]],
            gamma_range[min_idx[2]],
            sse_array[min_idx])

# helper that steps through a time‑series and allows parameter/compartment modifications at each day
def simulate_piecewise(beta, sigma, gamma,
                       S0, E0, I0, R0, N, days, interventions):
    """Return S,E,I,R arrays of length days+1.

    `interventions` is a list of callables taking
    (t,beta,gamma,Sprev,Eprev,Iprev,Rprev) and returning
    possibly modified values; called with t=1..days.
    """
    S = np.zeros(days+1); E = np.zeros(days+1)
    I = np.zeros(days+1); R = np.zeros(days+1)
    S[0], E[0], I[0], R[0] = S0, E0, I0, R0
    for t in range(1, days+1):
        b = beta
        g = gamma
        Spr, Epr, Ipr, Rpr = S[t-1], E[t-1], I[t-1], R[t-1]
        for interv in interventions:
            b, g, Spr, Epr, Ipr, Rpr = interv(t, b, g, Spr, Epr, Ipr, Rpr)
        dS, dE, dI, dR = seir_model((Spr, Epr, Ipr, Rpr),
                                    t-1, b, sigma, g, N)
        S[t] = Spr + dS
        E[t] = Epr + dE
        I[t] = Ipr + dI
        R[t] = Rpr + dR
    return S, E, I, R

# interventions (all start on day 70 unless otherwise noted)
def mask_intervention(start_day=70, reduction=0.4):
    return lambda t,b,g,S,E,I,R: (b*(1-reduction) if t>=start_day else b,
                                  g, S, E, I, R)

def vacc_campaign(day=70, n_vacc=2000, eff=0.9):
    def f(t,b,g,S,E,I,R):
        if t == day:
            vac = min(n_vacc, S) * eff
            S -= vac; R += vac
        return b, g, S, E, I, R
    return f

def vacc_rollout(days=[70,80,90], n_each=1000, eff=0.9):
    def f(t,b,g,S,E,I,R):
        if t in days:
            vac = min(n_each, S) * eff
            S -= vac; R += vac
        return b, g, S, E, I, R
    return f

def testing_quarantine(start_day=70, reduction_days=2):
    def f(t,b,g,S,E,I,R):
        if t > start_day:
            g = 1.0 / (1.0/g - reduction_days)   # shorten infectious
        return b, g, S, E, I, R
    return f

def school_closure(start_day=70, duration=14, contact_factor=0.2):
    return lambda t,b,g,S,E,I,R: (b*contact_factor if start_day <= t < start_day+duration else b,
                                  g, S, E, I, R)

# fit the UVA data (release #2) to obtain β,σ,γ for the first 70 days
df_uva = pd.read_csv("Data/mystery_virus_daily_active_counts_RELEASE#2.csv",
                     parse_dates=['date'])
obs_uva = df_uva["active reported daily cases"].values
t_uva   = np.arange(len(obs_uva))

# UVA initial conditions (same as before)
S0_uva = 17900
E0_uva = 0            # E0(VT) is set equal to this later
I0_uva = 1
R0_uva = 1.24
N_uva  = S0_uva + E0_uva + I0_uva + R0_uva

best_beta, best_sigma, best_gamma, best_sse = grid_search_parameters(
    t_uva, N_uva, S0_uva, E0_uva, I0_uva, R0_uva, obs_uva
)
print(f"UVA fit: β={best_beta:.3f}, σ={best_sigma:.3f}, "
      f"γ={best_gamma:.3f}, SSE={best_sse:.0f}")

# VT setup
S0_vt = 38900           # VT student population
E0_vt = E0_uva          # same exposed as UVA
I0_vt = 1
R0_vt = 0
N_vt  = S0_vt + E0_vt + I0_vt + R0_vt

horizon = 120           # total days to simulate

# baseline (no intervention)
S_base, E_base, I_base, R_base = simulate_piecewise(
    best_beta, best_sigma, best_gamma,
    S0_vt, E0_vt, I0_vt, R0_vt, N_vt,
    horizon, interventions=[]
)

def metrics(I_array, t0=70, t1=120):
    segment = I_array[t0:t1+1]
    return segment.max(), segment.sum()

base_peak, base_total = metrics(I_base)
print(f"baseline peak (70‑120) = {base_peak:.0f}, total = {base_total:.0f}")

# run every intervention
scenarios = {
    "mask mandate"       : [mask_intervention()],
    "vaccine campaign"   : [vacc_campaign()],
    "vaccine rollout"    : [vacc_rollout()],
    "testing + quarantine": [testing_quarantine()],
    "school closure"     : [school_closure()],
}

results = {}
for name, intervs in scenarios.items():
    S_i, E_i, I_i, R_i = simulate_piecewise(
        best_beta, best_sigma, best_gamma,
        S0_vt, E0_vt, I0_vt, R0_vt, N_vt,
        horizon, intervs
    )
    peak, total = metrics(I_i)
    results[name] = (I_i, peak, total)
    print(f"{name:20s} peak={peak:.0f}, total={total:.0f}")

# choose best intervention by smallest peak (or modify criterion)
best_name = min(results.items(), key=lambda kv: kv[1][1])[0]
print(f"\nrecommended intervention (lowest peak): {best_name}")

# plot the curves
plt.figure(figsize=(10,6))
plt.plot(np.arange(horizon+1), I_base, label='baseline', lw=2)
for name,(I_i,_,_) in results.items():
    plt.plot(np.arange(horizon+1), I_i, label=name)
plt.axvline(70, color='k', ls='--', label='intervention start')
plt.xlabel('days')
plt.ylabel('infected')
plt.title('VT epidemic, days 0–120')
plt.legend()
plt.grid()
plt.show()