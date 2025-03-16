import pandas as pd
import numpy as np

# Load datset
df = pd.read_csv('bloodtypes.csv')

# Entropy calculation
def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

binned_O_plus = pd.cut(df['O+'], bins=4, labels=False)
print("Entropy of O+:", entropy(binned_O_plus))
