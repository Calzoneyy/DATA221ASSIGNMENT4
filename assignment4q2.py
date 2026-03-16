from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# setup the data
data = load_breast_cancer()
X, y = data.data, data.target

# 80/20 train-test split with stratification so the class ratio stays the same
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

# train the unconstrained tree using entropy
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_entropy.fit(X_train, y_train)

# check the accuracy
train_acc = accuracy_score(y_train, dt_entropy.predict(X_train))
test_acc = accuracy_score(y_test, dt_entropy.predict(X_test))

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# --- Discussion ---
# What is entropy here?
# Entropy is basically a measure of messiness or randomness in the data. The decision tree
# tries to make splits that drop this entropy as much as possible so it can make confident predictions.

# Overfitting or good generalization?
# Definitely overfitting. The training accuracy is a perfect 1.0 (100%), meaning it memorized
# the training data. The test accuracy is lower, showing it doesn't generalize perfectly to new data.