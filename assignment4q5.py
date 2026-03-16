from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# setting up both models to compare them
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.20, stratify=data.target, random_state=42)

# train the tree
dt = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
dt.fit(X_train, y_train)

# train the NN
scaler = StandardScaler()
nn = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
nn.fit(scaler.fit_transform(X_train), y_train)

# let's look at the confusion matrices
print("Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, dt.predict(X_test)))

print("\nNeural Network Confusion Matrix:")
print(confusion_matrix(y_test, nn.predict(scaler.transform(X_test))))

# --- Discussion ---
# Which model would I prefer?
# For a medical diagnosis task, I'd go with the Decision Tree. Even if the neural net might
# edge it out slightly in accuracy, we need accountability. A doctor needs to be able to explain
# why a prediction was made, which a tree allows us to do.
#
# Pros and cons:
# Decision Tree: Awesome for interpretability (pro), but can be unstable and prone to overfitting if not tuned well (con).
# Neural Network: Great at finding complex, hidden patterns (pro), but acts as a total black box making it hard to trust without blind faith (con).