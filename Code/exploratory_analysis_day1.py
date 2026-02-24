#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Load the data
data = pd.read_csv("Data/mystery_virus_daily_active_counts_RELEASE#1.csv", parse_dates=['date'], header=0, index_col=None)

#%%
# Make a plot of the active cases over time
plt.figure(figsize=(10, 6))
plt.plot(data['date'], data['active reported daily cases'], marker='o', linestyle='-')
plt.title('Daily Active Cases of the Mystery Virus')
plt.xlabel('Date')
plt.ylabel('Number of Active Cases')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

'''
What do you notice about the initial infections?
The initial infections increase over time. The rate of increase accelerates, so the virus is spreading faster and faster. The curve is exponential.
How could we measure how quickly its spreading?
We could measure the rate of increase in the number of active cases over time. This can be done by calculating the growth rate, which is the percentage increase in active cases from one day to the next.
What information about the virus would be helpful in determining the shape of the outbreak curve?
Information about the virus's transmission rate, incubation period, and recovery time would be helpful in determining the shape of the outbreak curve.
'''

# day 2 plot
#estimate R0 for mystery virus data using a fit to the exponential growth in I
from scipy.optimize import curve_fit
import numpy as np
# Define the exponential growth function
def exponential_growth(t, I0, r):
    return I0 * np.exp(r * t)
# Prepare the data for fitting
data['days'] = (data['date'] - data['date'].min()).dt.days
# Fit the exponential growth model to the data
popt, pcov = curve_fit(exponential_growth, data['days'], data['active reported daily cases'], p0=(1, 0.1))
I0, r = popt
R0 = 1 + r * 2  # average infectious period of 2 days
print(f"Estimated R0: {R0:.2f}")
# Plot the data and the fitted curve
plt.figure(figsize=(10, 6))
plt.scatter(data['date'], data['active reported daily cases'], label='Data', color='blue')
t_fit = np.linspace(0, data['days'].max(), 100)
plt.plot(data['date'].min() + pd.to_timedelta(t_fit, unit='D'), exponential_growth(t_fit, *popt), label='Fitted Curve', color='red')
plt.title('Daily Active Cases of the Mystery Virus with Exponential Fit')
plt.xlabel('Date')
plt.ylabel('Number of Active Cases')
plt.xticks(rotation=45)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()