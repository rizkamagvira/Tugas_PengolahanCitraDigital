import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('koin2.jpeg') 

def adjust_brightness(image, brightness_factor):
    image = image.astype(np.float32)
    bright_image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
    return bright_image

def sharpen_image(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def adjust_contrast(image, contrast_factor):
    image = image.astype(np.float32)
    mean = np.mean(image)
    contrast_image = mean + contrast_factor * (image - mean)
    contrast_image = np.clip(contrast_image, 0, 255).astype(np.uint8)
    return contrast_image

# Menajamkan dan menyesuaikan kecerahan, kontras, dan menajamkan gambar
contrast_factor = 1

image_sharp = sharpen_image(img)
image_bright = adjust_brightness(image_sharp, 3)
image_contrast = adjust_contrast(image_bright, contrast_factor)

# Mengubah gambar ke grayscale
image_gray = cv2.cvtColor(image_contrast, cv2.COLOR_BGR2GRAY)

# Mengubah ke hitam & putih 
(thresh, blackAndWhiteImage) = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)

# Menghilangkan noise
bilateral = cv2.bilateralFilter(image_gray, 5, 100, 100)

# Menerapkan deteksi tepi Canny
canny = cv2.Canny(bilateral, 100, 200)

# Melakukan dilasi pada tepi untuk menutup celah
dilated = cv2.dilate(canny, (3, 3), iterations=2)
# Menemukan kontur
contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Menyaring kontur kecil
min_area = 500  # Sesuaikan nilai ini untuk menetapkan area minimum untuk sebuah kontur yang valid
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

# Menghitung jumlah koin
num_coins = len(filtered_contours)

# Mengonversi gambar Canny ke RGB untuk visualisasi
rgb = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
cv2.drawContours(rgb, filtered_contours, -1, (255, 0, 255), 2)  # Mengubah warna menjadi ungu (255, 0, 255)

# Gambar asli
plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Tepi Canny dengan jumlah koin
plt.subplot(1, 2, 2)
plt.title(f'Canny - Number of coins: {num_coins}')
plt.imshow(rgb)
plt.axis('off')
plt.show()


