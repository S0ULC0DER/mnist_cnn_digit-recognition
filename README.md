# MNIST CNN Digit Recognition

A handwritten digit recognition project built using Python, TensorFlow/Keras, NumPy, OpenCV, and Matplotlib.

The project uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize handwritten digits from 0 to 9.

## Project Overview

The project has two main parts:

1. Training a CNN on the MNIST dataset.
2. Using the trained model to recognize a handwritten digit from an external image.

The external image is processed to make it similar to the MNIST input format before being passed to the CNN.

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* OpenCV
* Matplotlib

## CNN Architecture

The model consists of:

* Conv2D - 32 filters
* MaxPooling2D
* Conv2D - 64 filters
* MaxPooling2D
* Flatten
* Dense - 128 neurons
* Dense - 10 neurons with Softmax

The final layer produces probabilities for the 10 possible digits (0-9).

## Image Preprocessing

External handwritten images may not have the same size or positioning as MNIST images.

The preprocessing pipeline is:

```text
Input Image
     ↓
Grayscale
     ↓
Invert Colors
     ↓
Find Non-Zero Pixels
     ↓
Find Bounding Rectangle
     ↓
Crop Digit
     ↓
Resize While Maintaining Aspect Ratio
     ↓
Center on 28 × 28 Canvas
     ↓
Normalize Pixel Values
     ↓
Add Channel and Batch Dimensions
     ↓
CNN Prediction
```

### Why preprocessing is required

The CNN was trained using 28 × 28 MNIST images. Therefore, an external handwritten digit needs to be converted into a similar format before prediction.

For example, the original handwritten image may contain a large amount of empty space or the digit may be positioned away from the center. The preprocessing step crops the digit and centers it on a 28 × 28 canvas.

## Manual CNN Implementation

As part of understanding how CNNs work internally, a simplified CNN pipeline was also implemented using NumPy.

It includes:

* Convolution
* ReLU activation
* Max pooling

This helped demonstrate how filters create feature maps and how pooling reduces their spatial dimensions.

## What I Learned

Through this project, I practiced:

* Loading and preprocessing image datasets
* Pixel normalization
* Understanding image dimensions and channels
* Convolution and feature maps
* ReLU activation
* Max pooling
* Flattening feature maps
* Dense layers
* Softmax classification
* CNN parameter calculations
* Model training and evaluation
* Image preprocessing using OpenCV
* Preparing external images for model inference

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd mnist-cnn-digit-recognition
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python src/train_model.py
```

The trained model will be saved in the `models` directory.

### 4. Predict a handwritten digit

Place an image inside:

```text
sample_images/
```

Then run:

```bash
python src/predict_digit.py
```

The processed image and predicted digit will be displayed.

## Example

A handwritten image containing the digit `5` is processed, centered, normalized, and passed to the CNN.

Example output:

```text
Input shape: (1, 28, 28, 1)
Predicted digit: 5
```

## Project Goal

The goal of this project was not only to build a digit classifier, but also to understand the basic concepts behind CNNs and how image preprocessing affects model predictions.
