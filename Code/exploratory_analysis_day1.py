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