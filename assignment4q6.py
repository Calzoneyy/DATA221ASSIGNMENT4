import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# grab the fashion mnist images
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# normalize the pixels to be between 0 and 1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# reshape to add the channel dimension (since they are grayscale)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# build the cnn
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train it for 15 epochs
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# check how it did on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# --- Discussion ---
# Why use a CNN instead of a normal fully connected network?
# Normal networks force you to flatten the image immediately, losing all the spatial layout.
# CNNs keep the 2D structure, allowing them to look for shapes and patterns in localized areas.
#
# What is the conv layer actually learning?
# It's learning to act like visual filters. At first, it picks up basic stuff like straight edges
# or curves, and as it gets deeper, it learns combinations of those to recognize things like sleeves or collars.