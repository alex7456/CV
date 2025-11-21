import cv2
import numpy as np

def process_image(path):
    # Считываем изображение
    img = cv2.imread(path)
    
    
    # Перевод в черно-белый формат
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Размытие по Гауссу
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # --- Вычисление градиентов Собеля ---
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)  # производная по X
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)  # производная по Y

    # --- Длина градиента ---
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # --- Угол градиента (в радианах) ---
    angle = np.arctan2(grad_y, grad_x)

    # Вывод матриц в консоль
    print("Матрица длин градиента:\n", magnitude)
    print("\nМатрица углов градиента (в радианах):\n", angle)

    # Нормализация для отображения
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ang_norm = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Вывод окон
    cv2.imshow("Черно-белое", gray)
    cv2.imshow("Градиенты - величина", mag_norm)
    cv2.imshow("Градиенты - угол", ang_norm)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

process_image("C:/Users/Alexandr/Desktop/cvision/4/img/2.png")
