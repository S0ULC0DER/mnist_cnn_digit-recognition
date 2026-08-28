import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)
print("Testing images:", x_test.shape)
print("Testing labels:", y_test.shape)


# Display a sample image
plt.imshow(x_train[0], cmap="gray")
plt.title(f"Label: {y_train[0]}")
plt.show()


# Normalize pixel values from 0-255 to 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0


# Add channel dimension
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]


# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dense(10, activation="softmax")
])


# Display model architecture
model.summary()


# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Train model
model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.1
)


# Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test)


print("Test accuracy:", test_accuracy)

# Save model
model.save("models/mnist_cnn.keras")
print("Model saved successfully.")
