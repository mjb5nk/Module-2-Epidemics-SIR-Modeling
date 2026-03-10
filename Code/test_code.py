import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the data for RELEASE#2
data = pd.read_csv("Data/mystery_virus_daily_active_counts_RELEASE#2.csv", parse_dates=['date'], header=0, index_col=None)
observed_data = data["active reported daily cases"].values
num_days = len(observed_data)
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

# Initial conditions
S0 = 17900
E0 = 0
I0 = 1
R0 = 1.24  # estimated from data (day 1 data R0 value)
N = S0 + E0 + I0 + R0

# Example parameters (will be fitted later)
beta = 0.6
sigma = 0.2
gamma = 0.2

# Solve the SEIR model
S, E, I, R = euler_method(beta, sigma, gamma, S0, E0, I0, R0, timepoints, N)

# Plot the SEIR model results
plt.figure(figsize=(10, 6))
plt.plot(timepoints, S, label='Susceptible')
plt.plot(timepoints, E, label='Exposed')
plt.plot(timepoints, I, label='Infected')
plt.plot(timepoints, R, label='Recovered')
plt.title('SEIR Model Simulation')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend()
plt.grid()
plt.show()



# Plot the VT data and the SEIR model fit
# Load the data for RELEASE#3
data = pd.read_csv("Data/mystery_virus_daily_active_counts_RELEASE#3.csv", parse_dates=['date'], header=0, index_col=None)
observed_data = data["active reported daily cases"].values
num_days = len(observed_data)
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

# Initial conditions
S0 = 17900
E0 = 0
I0 = 1
R0 = 1.24  # estimated from data (day 1 data R0 value)
N = S0 + E0 + I0 + R0

# Example parameters (will be fitted later)
beta = 0.6
sigma = 0.08
gamma = 0.2

# Solve the SEIR model
S, E, I, R = euler_method(beta, sigma, gamma, S0, E0, I0, R0, timepoints, N)

# Plot the SEIR model results
plt.figure(figsize=(10, 6))
plt.plot(timepoints, S, label='Susceptible')
plt.plot(timepoints, E, label='Exposed')
plt.plot(timepoints, I, label='Infected')
plt.plot(timepoints, R, label='Recovered')
plt.title('VT SEIR Model Simulation')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend()
plt.grid()
plt.show()


# Plot the VT data versus the originial SEIR model
plt.figure(figsize=(10, 6))
plt.plot(timepoints, observed_data, label='Observed Data', color='blue')
plt.plot(timepoints, I, label='SEIR Model Fit', color='red')
plt.title('SEIR Model Fit to VT Data')
plt.xlabel('Time (days)')
plt.ylabel('Number of Active Cases')
plt.legend()
plt.grid()
plt.show()


