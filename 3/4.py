import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma):
    """Создает ядро Гаусса заданного размера и отклонения."""
    k = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)

    for i in range(size):
        for j in range(size):
            x, y = i - k, j - k
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Нормировка (сумма всех элементов = 1)
    kernel /= np.sum(kernel)
    return kernel

def apply_gaussian_filter(image, size, sigma):
    """Применяет Гауссов фильтр вручную."""
    kernel = gaussian_kernel(size, sigma)
    k = size // 2
    h, w = image.shape
    new_img = np.zeros_like(image, dtype=np.float32)

    # Свертка вручную
    for i in range(k, h - k):
        for j in range(k, w - k):
            region = image[i - k:i + k + 1, j - k:j + k + 1]
            new_img[i, j] = np.sum(region * kernel)
    
    return np.uint8(new_img)

# ====== Проверка на изображении ======
# Загружаем изображение
img = cv2.imread('img/1.jpg', cv2.IMREAD_GRAYSCALE)
img_small = cv2.resize(img, (200, 200))  # уменьшение для теста
# Применяем разные параметры
res1 = apply_gaussian_filter(img_small, size=3, sigma=1)
res2 = apply_gaussian_filter(img_small, size=7, sigma=1)
res3 = apply_gaussian_filter(img_small, size=7, sigma=2)

# Отображаем результаты
plt.figure(figsize=(12,6))
plt.subplot(1,4,1), plt.imshow(img, cmap='gray'), plt.title('Исходное')
plt.subplot(1,4,2), plt.imshow(res1, cmap='gray'), plt.title('3x3, σ=1')
plt.subplot(1,4,3), plt.imshow(res2, cmap='gray'), plt.title('7x7, σ=1')
plt.subplot(1,4,4), plt.imshow(res3, cmap='gray'), plt.title('7x7, σ=2')
plt.show()