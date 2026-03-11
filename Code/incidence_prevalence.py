# calculate the incidence and prevalence of a disease in a population
# Population size is 17900
# Have to use Data set 2 because this is UVA population only
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Load the data for RELEASE#2
data = pd.read_csv("GitHub/module-2-mjb5nk/Module-2-Epidemics-SIR-Modeling/Data/mystery_virus_daily_active_counts_RELEASE#2.csv", parse_dates=['date'], header=0, index_col=None)
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
plt.savefig('GitHub/module-2-mjb5nk/Module-2-Epidemics-SIR-Modeling/Notebook examples/incidence_prevalence.png', dpi=300, bbox_inches='tight')
plt.show()

# Find highest and lowest incidence
pop = 17900
incidence_pct = (incidence / pop) * 100

max_incidence_idx = np.argmax(incidence_pct)
min_incidence_idx = np.argmin(incidence_pct)

max_incidence_day = data.iloc[max_incidence_idx]['date']
max_incidence_pct = incidence_pct[max_incidence_idx]

min_incidence_day = data.iloc[min_incidence_idx]['date']
min_incidence_pct = incidence_pct[min_incidence_idx]

print(f"\nHighest Incidence:")
print(f"  Day: {max_incidence_day} (Day {max_incidence_idx + 1})")
print(f"  Incidence: {max_incidence_pct:.4f}%")

print(f"\nLowest Incidence:")
print(f"  Day: {min_incidence_day} (Day {min_incidence_idx + 1})")
print(f"  Incidence: {min_incidence_pct:.4f}%")
# Find highest and lowest prevalence
prevalence_pct = (prevalence / pop) * 100

max_prevalence_idx = np.argmax(prevalence_pct)
min_prevalence_idx = np.argmin(prevalence_pct)

max_prevalence_day = data.iloc[max_prevalence_idx]['date']
max_prevalence_pct = prevalence_pct[max_prevalence_idx]

min_prevalence_day = data.iloc[min_prevalence_idx]['date']
min_prevalence_pct = prevalence_pct[min_prevalence_idx]

print(f"\nHighest Prevalence:")
print(f"  Day: {max_prevalence_day} (Day {max_prevalence_idx + 1})")
print(f"  Prevalence: {max_prevalence_pct:.4f}%")

print(f"\nLowest Prevalence:")
print(f"  Day: {min_prevalence_day} (Day {min_prevalence_idx + 1})")
print(f"  Prevalence: {min_prevalence_pct:.4f}%")