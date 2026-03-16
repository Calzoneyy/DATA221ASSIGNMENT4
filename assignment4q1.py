import numpy as np
from sklearn.datasets import load_breast_cancer

# load up the dataset
data = load_breast_cancer()

# grab the features and the target labels
X = data.data
y = data.target

# let's see the shapes
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# figure out how many samples we have for benign vs malignant
unique, counts = np.unique(y, return_counts=True)
class_distribution = dict(zip(data.target_names, counts))
print("Class distribution:", class_distribution)

# --- Discussion ---
# Is the dataset balanced?
# It's a bit imbalanced. We have 357 benign samples and 212 malignant ones.

# Why do we care about class balance?
# If the data is super imbalanced, the model might just get lazy and predict the majority
# class all the time. In a medical setting, missing a malignant case (the minority class)
# is really dangerous, so we need to make sure the model is actually learning to spot it.