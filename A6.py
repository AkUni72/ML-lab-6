from sklearn.tree import plot_tree, DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv('bloodtypes.csv')

# Prepare data
X = df.drop(['Country'], axis=1)
y = pd.cut(df['O+'], bins=4, labels=False)

# Train decision tree
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3)
tree.fit(X, y)

# Plot tree
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=X.columns, filled=True)
plt.show()
