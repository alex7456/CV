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


# Применяем фильтр
result = apply_gaussian_filter(img_small, size=11, sigma=3.0)

# Отображаем результат
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Исходное')

plt.subplot(1, 2, 2)
plt.imshow(result, cmap='gray')
plt.title('После фильтра Гаусса')

plt.show()
