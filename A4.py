import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('bloodtypes.csv')

# Binning function
def binning(data, num_bins=4):
    bins = np.linspace(data.min(), data.max(), num_bins + 1)
    return np.digitize(data, bins) - 1

binned_population = binning(df['Population'], num_bins=4)
print("Binned Population:", binned_population)
