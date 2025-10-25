import numpy as np

def gaussian_kernel(size, sigma):
    """Создает ядро Гаусса заданного размера и отклонения."""
    k = size // 2  # центр ядра
    kernel = np.zeros((size, size), dtype=np.float32)
    
    for i in range(size):
        for j in range(size):
            x, y = i - k, j - k
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Нормировка — сумма элементов = 1
    kernel /= np.sum(kernel)
    return kernel

# Проверим ядра для разных размеров
for s in [3, 5, 7]:
    print(f"\nЯдро Гаусса {s}x{s}:")
    print(gaussian_kernel(s, sigma=1))
