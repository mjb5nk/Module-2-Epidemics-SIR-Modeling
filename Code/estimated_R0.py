import pandas as pd
import matplotlib.pyplot as plt

#%%
# Load the data
data = pd.read_csv("Data/mystery_virus_daily_active_counts_RELEASE#1.csv", parse_dates=['date'], header=0, index_col=None)


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

'''
What viruses have a similar R0? Use the viruses.html file to find a virus or 2 with a similar R0 and give a 1-2 sentence background of the diseases.

The estimated R0 for the mystery virus is 1.24, which is similar to the R0 of seasonal influenza (R0 of 1.3). 
Seasonal influenza is a contagious respiratory illness caused by influenza viruses, which can lead to mild to severe illness and can sometimes result in death,
particularly in vulnerable populations such as the elderly and those with underlying health conditions.

This R0 is also similar to the R0 of Rhinovirus (R0 of 1.5).
Rhinovirus is a common viral infectious agent that primarily causes the common cold. 
It is highly contagious and can lead to symptoms such as a runny nose, sore throat, cough, and congestion. Rhinovirus infections are typically mild and self-limiting,
but they can cause complications in individuals with weakened immune systems or pre-existing respiratory conditions.

How accurate do you think your R0 estimate is?
The R0 estimate is based on the early exponential growth phase of the outbreak, which can be influenced by various factors such as underreporting of cases, changes in testing rates, and public health interventions. Therefore, while the estimate provides a useful measure of transmissibility, it may not be perfectly accurate and should be interpreted with caution.
'''