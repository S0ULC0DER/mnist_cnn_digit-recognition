import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Path to handwritten digit
path = r"sample_images/number5.jpg"


# Load image as grayscale
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Could not load image: {path}")


# Invert image
# Background becomes black and digit becomes white
img = 255 - img


# Find non-zero pixels
coords = cv2.findNonZero(img)

if coords is None:
    raise ValueError("No digit detected in the image.")


# Find bounding rectangle around digit
x, y, w, h = cv2.boundingRect(coords)


# Crop digit
digit = img[y:y+h, x:x+w]


# Resize while maintaining aspect ratio
scale = 20 / max(w, h)

new_w = int(w * scale)
new_h = int(h * scale)

digit = cv2.resize(digit, (new_w, new_h))


# Create 28x28 black canvas
canvas = np.zeros((28, 28), dtype=np.uint8)


# Calculate offsets to center digit
x_offset = (28 - new_w) // 2
y_offset = (28 - new_h) // 2


# Place digit in center
canvas[
    y_offset:y_offset + new_h,
    x_offset:x_offset + new_w
] = digit


# Use centered image as input
img = canvas


# Display processed image
plt.imshow(img, cmap="gray")
plt.title("Processed Digit")
plt.show()


# Normalize pixel values
img = img / 255.0


# Add channel dimension
img = img[..., np.newaxis]


# Add batch dimension
img = img[np.newaxis, ...]


print("Input shape:", img.shape)


# Load trained model
model = tf.keras.models.load_model("models/mnist_cnn.keras")


# Make prediction
prediction = model.predict(img)


# Get digit with highest probability
digit = np.argmax(prediction[0])


print("Predicted digit:", digit)
