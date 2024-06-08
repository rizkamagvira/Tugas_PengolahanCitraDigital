import cv2
import numpy as np
import matplotlib.pyplot as plt

#Default
# def sharpen_image(image):
#     kernel = np.array([[-1, -1, -1],
#                        [-1, 9, -1],
#                        [-1, -1, -1]])
#     return cv2.filter2D(image, -1, kernel)

# Uncomment these lines to test with different kernels
# Test 1: 3x3 averaging kernel
# def sharpen_image(image):
#     kernel = np.array([[1/9, 1/9, 1/9],
#                        [1/9, 1/9, 1/9],
#                        [1/9, 1/9, 1/9]])
#     return cv2.filter2D(image, -1, kernel)

# Test 2: 5x5 averaging kernel
def sharpen_image(image):
    kernel = np.array([[1/25, 1/25, 1/25, 1/25, 1/25],
                       [1/25, 1/25, 1/25, 1/25, 1/25],
                       [1/25, 1/25, 1/25, 1/25, 1/25],
                       [1/25, 1/25, 1/25, 1/25, 1/25],
                       [1/25, 1/25, 1/25, 1/25, 1/25]])
    return cv2.filter2D(image, -1, kernel)

# Read the image
image = cv2.imread('sample.jpeg')


if image is None:
    print("Error: Could not open or find the image.")
else:

    sharpened_image = sharpen_image(image)


    original_and_sharpened_image = np.hstack((image, sharpened_image))


    plt.figure(figsize=(20, 10))
    plt.axis('off')
    plt.imshow(cv2.cvtColor(original_and_sharpened_image, cv2.COLOR_BGR2RGB))
    plt.show()
