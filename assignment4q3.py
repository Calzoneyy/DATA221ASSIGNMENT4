import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# setup the data again
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.20, stratify=data.target, random_state=42)

# add some constraints so it doesn't just memorize the data
dt_constrained = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_split=5, random_state=42)
dt_constrained.fit(X_train, y_train)

# get the new accuracies
print(f"Constrained Training Accuracy: {accuracy_score(y_train, dt_constrained.predict(X_train)):.4f}")
print(f"Constrained Test Accuracy: {accuracy_score(y_test, dt_constrained.predict(X_test)):.4f}")

# what features is the model actually looking at?
importances = pd.Series(dt_constrained.feature_importances_, index=data.feature_names)
print("\nTop 5 Most Important Features:")
print(importances.sort_values(ascending=False).head(5))

# --- Discussion ---
# How do constraints affect overfitting?
# By limiting the max depth, we stop the tree from growing endlessly and making hyperspecific
# rules for every single training point. It forces the model to learn the general patterns instead.
#
# Feature importance and interpretability:
# Feature importance tells us exactly which metrics (like worst perimeter or worst area) the
# model cares about most. This makes the tree super easy to interpret and explain to a doctor.