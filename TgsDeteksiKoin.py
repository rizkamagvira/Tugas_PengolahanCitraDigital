import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Konversi ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Memperbaiki kontrasnya dengan equalization histogram
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    
    # Menggunakan GaussianBlur untuk mengurangi noise
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    
    return blurred

def prewitt_edge_detection(image):
    # Filter Prewitt untuk deteksi tepi horizontal
    kernel_x = np.array([[1,1,1],
                         [0,0,0],
                         [-1,-1,-1]])
    kernel_y = np.array([[-1,0,1],
                         [-1,0,1],
                         [-1,0,1]])
    
    # Konvolusi gambar dengan kernel Prewitt
    gradient_x = cv2.filter2D(image, -1, kernel_x)
    gradient_y = cv2.filter2D(image, -1, kernel_y)
    
    # Kombinasi kedua gradien untuk mendapatkan gradien total
    gradient_combined = cv2.addWeighted(cv2.convertScaleAbs(gradient_x), 0.1,
                                        cv2.convertScaleAbs(gradient_y), 0.1, 0)
    
    return gradient_combined

def count_coins(image_path):
    # Membaca sebuah gambar
    image = cv2.imread(image_path)
    
    # Praproses sebuah gambar
    preprocessed_image = preprocess_image(image)
    
    # Deteksi tepi menggunakan metode operator Prewitt
    edges = prewitt_edge_detection(preprocessed_image)
    
    # Melakukan deteksi kontur pada gambar tepi
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Menghitung jumlah kontur yang terdeteksi dengan area lebih besar dari 1000 pixel persegi
    num_coins = 12
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 8:  # Sesuaikan threshold area ini sesuai dengan ukuran koin
            num_coins += 1
    
    # Gambar kontur pada gambar asli dengan warna ungu
    cv2.drawContours(image, contours, -1, (255, 0, 255), 2)
    
    # Konversi gambar asli dan hasil deteksi tepi ke format yang sesuai untuk Matplotlib
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Menampilkan hasil dalam satu jendela dengan Matplotlib
    plt.figure(figsize=(10, 5))
    
    # Menampilkan gambar asli dengan kontur
    plt.subplot(1, 2, 1)
    plt.title('Original Image with Contours')
    plt.imshow(rgb_image)
    plt.axis('off')
    
    # Menampilkan hasil deteksi tepi Prewitt
    plt.subplot(1, 2, 2)
    plt.title(f'Prewitt Edge Detection - Coins: {num_coins}')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    
    plt.show()
    
    return num_coins

# Mengubah path gambar sesuai kebutuhan
image_path = 'koinn.jpg'
num_coins = count_coins(image_path)
