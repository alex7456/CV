import numpy as np

def gaussian_kernel(size, sigma):
    k = size // 2  
    kernel = np.zeros((size, size), dtype=np.float32)

    
    a, b = 0, 0  

    for i in range(size):
        for j in range(size):
            x, y = i - k, j - k
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - a)**2 + (y - b)**2) / (2 * sigma**2))

    return kernel



kernel_3x3 = gaussian_kernel(3, 1)
print("Ядро Гаусса 3x3 (σ=1) без нормировки:")
print(kernel_3x3)

print("\nСумма элементов ядра:", np.sum(kernel_3x3))
