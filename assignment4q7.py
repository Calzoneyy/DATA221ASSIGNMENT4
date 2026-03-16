import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import confusion_matrix

# Note: this script assumes you have a trained model. I'm loading the data and
# doing a quick train here just so this script runs standalone without crashing.
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# FIXED LINE: Slice the array first, then cast and divide
X_train = X_train[:1000].astype('float32') / 255.0
y_train = y_train[:1000]

X_test = X_test.astype('float32') / 255.0
X_test_reshaped = X_test.reshape(-1, 28, 28, 1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X_train.reshape(-1, 28, 28, 1), y_train, epochs=1, verbose=0) # quick dummy train

# actual analysis starts here
predictions = model.predict(X_test_reshaped)
y_pred_classes = np.argmax(predictions, axis=1)

print("CNN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

# find where the model messed up
misclassified = np.where(y_pred_classes != y_test)[0]
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plot 3 mistakes
plt.figure(figsize=(10, 4))
for i, idx in enumerate(misclassified[:3]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(X_test[idx], cmap='gray')
    plt.title(f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred_classes[idx]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# --- Discussion ---
# Patterns in the errors:
# The model tends to mix up classes that have really similar outlines or silhouettes.
# For example, it often gets confused between a 'Shirt' and a 'T-shirt/top' because they occupy
# roughly the same space and shape in a low-res 28x28 image.
#
# How to fix it:
# A realistic fix would be using data augmentation. If we randomly flip, zoom, or slightly rotate
# the images during training, the model is forced to learn more robust, distinct features rather
# than relying on the exact pixel placement of a shirt.