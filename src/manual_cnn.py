import numpy as np


# Convolution operation
def convolve(image, kernel):

    output_height = image.shape[0] - kernel.shape[0] + 1
    output_width = image.shape[1] - kernel.shape[1] + 1

    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):

            # Extract a small region of the image
            patch = image[
                i:i + kernel.shape[0],
                j:j + kernel.shape[1]
            ]

            # Element-wise multiplication
            result = patch * kernel

            # Sum the multiplied values
            conv_value = np.sum(result)

            output[i, j] = conv_value

    return output


# ReLU activation function
def relu(feature_map):
    return np.maximum(0, feature_map)


# Max pooling operation
def max_pool(feature_map, pool_size=2, stride=2):

    pool_output_height = (
        (feature_map.shape[0] - pool_size) // stride
    ) + 1

    pool_output_width = (
        (feature_map.shape[1] - pool_size) // stride
    ) + 1

    pool_output = np.zeros(
        (pool_output_height, pool_output_width)
    )

    for i in range(pool_output_height):
        for j in range(pool_output_width):

            # Extract pooling region
            patch = feature_map[
                i * stride:i * stride + pool_size,
                j * stride:j * stride + pool_size
            ]

            # Select maximum value
            pool_output[i, j] = np.max(patch)

    return pool_output


# Example input image
image = np.array([
    [0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 0, 1]
])


# Vertical edge detection filter
vertical_filter = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])


# Horizontal edge detection filter
horizontal_filter = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
])


# Apply convolution
vertical_output = convolve(image, vertical_filter)
horizontal_output = convolve(image, horizontal_filter)


# Apply ReLU
vertical_relu = relu(vertical_output)
horizontal_relu = relu(horizontal_output)


# Apply max pooling
vertical_pool = max_pool(vertical_relu)
horizontal_pool = max_pool(horizontal_relu)


# Display results
print("Vertical filter after max pooling:")
print(vertical_pool)

print("\nHorizontal filter after max pooling:")
print(horizontal_pool)
