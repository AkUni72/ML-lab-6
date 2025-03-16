import pandas as pd
import numpy as np

df = pd.read_csv('bloodtypes.csv')
X = df.drop(['Country'], axis=1).values
y = pd.cut(df['O+'], bins=4, labels=False)  

# Entropy calculation
def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

# Information Gain calculation
def information_gain(X, y, feature_index):
    parent_entropy = entropy(y)
    values, counts = np.unique(X[:, feature_index], return_counts=True)
    weighted_entropy = sum(
        (counts[i] / len(y)) * entropy(y[X[:, feature_index] == values[i]]) for i in range(len(values))
    )
    return parent_entropy - weighted_entropy

# Identify best root node
gains = [information_gain(X, y, i) for i in range(X.shape[1])]
best_feature = np.argmax(gains)
print("Best Root Feature Index:", best_feature)
