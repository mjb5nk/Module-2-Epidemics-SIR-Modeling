# calculate the incidence and prevalence of a disease in a population
# Population size is 17900
# Have to use Data set 2 because this is UVA population only
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Load the data for RELEASE#2
data = pd.read_csv("data/mystery_virus_daily_active_counts_RELEASE#2.csv", parse_dates=['date'], header=0, index_col=None)
observed_data = data["active reported daily cases"].values
num_days = len(observed_data)
timepoints = np.arange(num_days)
# Calculate incidence and prevalence
incidence = np.diff(observed_data, prepend=0)  # daily new cases
prevalence = observed_data  # active cases at each time point
# Plot incidence and prevalence
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(timepoints, incidence, label='Incidence (Daily New Cases)', color='blue')
plt.xlabel('Days')
plt.ylabel('Number of Cases')
plt.title('Incidence of the Disease')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(timepoints, prevalence, label='Prevalence (Active Cases)', color='orange')
plt.xlabel('Days')
plt.ylabel('Number of Cases')
plt.title('Prevalence of the Disease')
plt.legend()
plt.tight_layout()
plt.show()

# Do it a different way 

pop = 17900
base = "Data"   # adjust if your notebook is elsewhere

def load_and_calc(fname):
    df = pd.read_csv(f"{base}/{fname}")
    df["prevalence_pct"] = df["active reported daily cases"] / pop * 100
    # difference gives new cases; make first‑day equal to its count
    df["new_cases"] = df["active reported daily cases"].diff().fillna(
        df["active reported daily cases"]
    )
    df["incidence_pct"] = df["new_cases"] / pop * 100
    return df

df1 = load_and_calc("mystery_virus_daily_active_counts_RELEASE#1.csv")
df2 = load_and_calc("mystery_virus_daily_active_counts_RELEASE#2.csv")

print("dataset #1")
print(df1[["day","active reported daily cases",
           "prevalence_pct","incidence_pct"]].head())
print(df1[["day","active reported daily cases",
           "prevalence_pct","incidence_pct"]].tail())

print("\ndataset #2")
print(df2[["day","active reported daily cases",
           "prevalence_pct","incidence_pct"]].head())
print(df2[["day","active reported daily cases",
           "prevalence_pct","incidence_pct"]].tail())