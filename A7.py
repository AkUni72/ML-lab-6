import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv('bloodtypes.csv')

# Use two features for visualization
X = df[['O+', 'A+']].values
y = pd.cut(df['O+'], bins=4, labels=False)

# Train decision tree
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3)
tree.fit(X, y)

# Plot decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),np.linspace(y_min, y_max, 100))
Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="coolwarm", edgecolor="k")
plt.xlabel("O+")
plt.ylabel("A+")
plt.title("Decision Boundary")
plt.show()