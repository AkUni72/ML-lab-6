import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('bloodtypes.csv')

# Gini Index calculation
def gini_index(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return 1 - np.sum([p**2 for p in probabilities])

binned_O_plus = pd.cut(df['O+'], bins=4, labels=False)
print("Gini Index of O+:", gini_index(binned_O_plus))
