import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("Data/mystery_virus_daily_active_counts_RELEASE#2.csv", parse_dates=['date'], header=0, index_col=None)
# Extract observed infected counts
observed_data = data["active reported daily cases"].values
num_days = len(observed_data)

# Create matching timepoints for the observed data
timepoints = np.arange(num_days)

# SEIR differential equations
def seir_model(y, t, beta, sigma, gamma, N):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# Euler method implementation
def euler_method(beta, sigma, gamma, S0, E0, I0, R0, timepoints, N):
    S = np.zeros(len(timepoints))
    E = np.zeros(len(timepoints))
    I = np.zeros(len(timepoints))
    R = np.zeros(len(timepoints))
    # Initial conditions
    S[0], E[0], I[0], R[0] = S0, E0, I0, R0
    # Euler integration
    for i in range(1, len(timepoints)):
        dt = timepoints[i] - timepoints[i-1]
        dSdt, dEdt, dIdt, dRdt = seir_model(
            (S[i-1], E[i-1], I[i-1], R[i-1]),
            timepoints[i-1],
            beta, sigma, gamma, N
        )
        # Update values using Euler's method
        S[i] = S[i-1] + dSdt * dt
        E[i] = E[i-1] + dEdt * dt
        I[i] = I[i-1] + dIdt * dt
        R[i] = R[i-1] + dRdt * dt
    # return results
    return S, E, I, R

# Grid search for parameter estimation
def grid_search_parameters(timepoints, N,
                           S0, E0, I0, R0,
                           observed_data,
                           beta_range=None,
                           sigma_range=None,
                           gamma_range=None):
    # Define default parameter ranges if not provided
    if beta_range is None:
        beta_range = np.linspace(0.1, 1.0, 10)
    if sigma_range is None:
        sigma_range = np.linspace(0.1, 1.0, 10)
    if gamma_range is None:
        gamma_range = np.linspace(0.1, 1.0, 10)
    # Initialize array to store SSE values for each parameter combination
    sse_array = np.zeros((len(beta_range),
                          len(sigma_range),
                          len(gamma_range)))
    # Loop through all combinations of parameters and compute SSE
    for b, beta in enumerate(beta_range):
        for s, sigma in enumerate(sigma_range):
            for g, gamma in enumerate(gamma_range):
                S, E, I, R = euler_method(beta, sigma, gamma,
                                          S0, E0, I0, R0,
                                          timepoints, N)
                sse_array[b, s, g] = np.sum((I - observed_data) ** 2)
    # Find the parameter combination with the lowest SSE
    min_index = np.unravel_index(np.argmin(sse_array), sse_array.shape)
    best_beta = beta_range[min_index[0]]
    best_sigma = sigma_range[min_index[1]]
    best_gamma = gamma_range[min_index[2]]
    best_sse = sse_array[min_index]
    # return best parameters and SSE
    return best_beta, best_sigma, best_gamma, best_sse

# Inputs / initial conditions
S0 = 17900
E0 = 0
I0 = 1
R0 = 1.24 # estimated from data
N = S0 + E0 + I0 + R0

# perform grid search using the helper function
best_beta, best_sigma, best_gamma, best_sse = grid_search_parameters(
    timepoints, N, S0, E0, I0, R0, observed_data
)
# Print best parameters and SSE
print(f"Best parameters: beta={best_beta}, sigma={best_sigma}, "
      f"gamma={best_gamma}, SSE={best_sse}")

# Plot best-fit model
S, E, I, R = euler_method(best_beta, best_sigma, best_gamma, S0, E0, I0, R0, timepoints, N)
# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(timepoints, S, label='Susceptible')
plt.plot(timepoints, E, label='Exposed')
plt.plot(timepoints, I, label='Infected')
plt.plot(timepoints, R, label='Recovered')
plt.scatter(timepoints, observed_data, color='red', label='Observed Infected')
plt.title('SEIR Model Fit to Data')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend()
plt.grid()
plt.show()

# plot the full dataset (Data Release #3) against model
data_release_3 = pd.read_csv("Data/mystery_virus_daily_active_counts_RELEASE#3.csv",
                             parse_dates=['date'], header=0, index_col=None)
observed_data_release_3 = data_release_3["active reported daily cases"].values
timepoints_release_3 = np.arange(len(observed_data_release_3))

# run the model out to day 120 so the blue line extends that far
model_timepoints_120 = np.arange(120)
# re‑use best parameters / initial conditions
_, _, I_120, _ = euler_method(best_beta, best_sigma, best_gamma,
                              S0, E0, I0, R0,
                              model_timepoints_120, N)

# plot model predictions against observed data from Data Release #3
plt.figure(figsize=(10, 6))
plt.plot(model_timepoints_120, I_120,
         label='Model Infected (extended to day 120)')
plt.scatter(timepoints_release_3, observed_data_release_3,
            color='orange', label='Observed Infected (Release #3)')
plt.title('SEIR Model vs Data Release #3')
plt.xlabel('Time (days)')
plt.ylabel('Number of Infected Individuals')
plt.legend()
plt.grid()
plt.show()