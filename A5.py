from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Load dataset
df = pd.read_csv('bloodtypes.csv')

# Prepare data
X = df.drop(['Country'], axis=1)
y = pd.cut(df['O+'], bins=4, labels=False)

# Train decision tree
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3)
tree.fit(X, y)
print("Decision Tree Trained!")