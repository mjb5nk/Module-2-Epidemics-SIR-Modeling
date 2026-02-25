import pandas as pd
import matplotlib.pyplot as plt
#%%
# Load the data
data = pd.read_csv("Data/mystery_virus_daily_active_counts_RELEASE#1.csv", parse_dates=['date'], header=0, index_col=None)
#%%
# In a .py file (that will be copied to your final notebook): Estimate R0 for the mystery virus data using a fit to the exponential growth in I.
# We don’t know the transmission rate or recovery rate for the mystery virus, like we do in the game  we will estimate these parameters based on the data (working backwards from what we did in the game)


