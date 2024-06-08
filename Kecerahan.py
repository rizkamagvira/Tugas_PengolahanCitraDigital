import cv2
import numpy as np 
import matplotlib.pyplot as plt


def adjust_brightness(image, brightness_factor):

    image = image.astype(np.float32) 


    bright_image = image * brightness_factor


    bright_image = np.clip(bright_image, 0, 255).astype(np.uint8)

    return bright_image



image = cv2.imread('sample.jpeg') 

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# coba variasi lain sampe 3.0
brightness_factor = 1.9 
bright_image_rgb = adjust_brightness(image_rgb, brightness_factor) 


plt.figure(figsize=(10, 5)) 

plt.subplot(1, 2, 1) 
plt.title('Original Image') 
plt.imshow(image_rgb) 
plt.axis('off') 

plt.subplot(1, 2, 2) 
plt.title('Hasil')
plt.imshow(bright_image_rgb) 
plt.axis('off')

plt.show()