import numpy as np
from scipy.integrate import odeint

## inputs: beta, sigma, gamma, S0, E0, I0, R0, timepoints, N
beta = 0.3  # transmission rate
sigma = 0.2  # incubation rate
gamma = 0.1  # recovery rate
S0 = 999  # initial susceptible population
E0 = 1    # initial exposed population
I0 = 0    # initial infected population
R0 = 0    # initial recovered population
timepoints = np.linspace(0, 160, 160)  # time points for simulation
N = S0 + E0 + I0 + R0  # total population

# placeholder for observed data used when computing SSE
# replace this with your actual measurements for I(t)
observed_data = np.zeros_like(timepoints)

# initialize range for beta, sigma, and gamma
beta_range = np.linspace(0.1, 1.0, 10)  # transmission rate
sigma_range = np.linspace(0.1, 1.0, 10) # incubation rate
gamma_range = np.linspace(0.1, 1.0, 10)  # recovery rate

# initialize an empty array of SSE
sse_array = np.zeros((len(beta_range), len(sigma_range), len(gamma_range)))

# define euler method to solve SEIR model and calculate SSE for each combination of parameters

import numpy as np

def seir_model(y, t, beta, sigma, gamma, N):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

def euler_method(beta, sigma, gamma,
                 S0, E0, I0, R0,
                 timepoints, N):
    # allocate storage
    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))

    # set initial values
    S[0], E[0], I[0], R[0] = S0, E0, I0, R0

    # advance one step per timepoint
    for i in range(1, len(timepoints)):
        t_prev = timepoints[i-1]
        dt     = timepoints[i] - t_prev
        dSdt, dEdt, dIdt, dRdt = seir_model(
            (S[i-1], E[i-1], I[i-1], R[i-1]),
            t_prev, beta, sigma, gamma, N
        )

        S[i] = S[i-1] + dSdt * dt
        E[i] = E[i-1] + dEdt * dt
        I[i] = I[i-1] + dIdt * dt
        R[i] = R[i-1] + dRdt * dt

    return S, E, I, R

# makes arrays of values given each range for each parameter
for b, beta in enumerate(beta_range):
    for s, sigma in enumerate(sigma_range):
        for g, gamma in enumerate(gamma_range):
            # Run the SEIR model with the current parameters
            S, E, I, R = euler_method(beta, sigma, gamma, S0, E0, I0, R0, timepoints, N)
            # Calculate the SSE between the model predictions and the observed data
            sse = np.sum((I - observed_data) ** 2)
            sse_array[b, s, g] = sse

# print the parameter combination with the lowest SSE
min_index = np.unravel_index(np.argmin(sse_array), sse_array.shape)
best_beta = beta_range[min_index[0]]
best_sigma = sigma_range[min_index[1]]
best_gamma = gamma_range[min_index[2]]
print(f"Best parameters: beta={best_beta}, sigma={best_sigma}, gamma={best_gamma}")
