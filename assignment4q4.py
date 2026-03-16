from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# data setup
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.20, stratify=data.target, random_state=42)

# standardizing the features is a must for neural networks
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# building a simple neural net for binary classification
nn_model = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', solver='adam', max_iter=500, random_state=42)
nn_model.fit(X_train_scaled, y_train)

print(f"NN Training Accuracy: {accuracy_score(y_train, nn_model.predict(X_train_scaled)):.4f}")
print(f"NN Test Accuracy: {accuracy_score(y_test, nn_model.predict(X_test_scaled)):.4f}")

# --- Discussion ---
# Why do we need feature scaling for NNs?
# Neural nets use gradient descent to learn weights. If one feature is measured in thousands and
# another is a tiny decimal, the updates get completely thrown off and the model struggles to converge.
# Standardizing puts everything on the same playing field.
#
# What's an epoch?
# An epoch is just one complete runthrough of the entire training dataset during the learning process.