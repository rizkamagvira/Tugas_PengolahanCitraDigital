import cv2
import numpy as np
import matplotlib.pyplot as plt


def adjust_contrast(image, contrast_factor):

    image = image.astype(np.float32)
    mean = np.mean(image)
    contrast_image = mean + contrast_factor * (image - mean)
    contrast_image = np.clip(contrast_image, 0, 255).astype(np.uint8)
    return contrast_image



image = cv2.imread('sample.jpeg')

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 1 - 3.0
contrast_factor = 1.5
contrast_image_rgb = adjust_contrast(image_rgb, contrast_factor)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Hasil Kontras')
plt.imshow(contrast_image_rgb)
plt.axis('off')

plt.show()
