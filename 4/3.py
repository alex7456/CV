import cv2
import numpy as np

def non_max_suppression(magnitude, angle):
    H, W = magnitude.shape
    output = np.zeros((H, W), dtype=np.float64)

    # Перевод углов из радиан в градусы
    angle_deg = angle * 180.0 / np.pi
    angle_deg[angle_deg < 0] += 180  # диапазон 0..180

    for i in range(1, H - 1):
        for j in range(1, W - 1):

            q = 255
            r = 255

            # --- Определяем направление градиента ---
            a = angle_deg[i, j]

            # 0 градусов (по горизонтали)
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]

            # 45 градусов (↘ ↖)
            elif 22.5 <= a < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]

            # 90 градусов (по вертикали)
            elif 67.5 <= a < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]

            # 135 градусов (↗ ↙)
            elif 112.5 <= a < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            # --- Подавление немаксимумов ---
            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                output[i, j] = magnitude[i, j]
            else:
                output[i, j] = 0

    return output


def process_image(path):
    # Считывание изображения
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Гауссово размытие
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Градиенты
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x)

    # --- Подавление немаксимумов ---
    nms = non_max_suppression(magnitude, angle)

    # Нормализация для отображения
    nms_norm = cv2.normalize(nms, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Вывод
    cv2.imshow("Оригинал (серый)", gray)
    cv2.imshow("Градиенты - величина", mag_norm)
    cv2.imshow("После подавления немаксимумов", nms_norm)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


process_image("C:/Users/Alexandr/Desktop/cvision/4/img/2.png")
