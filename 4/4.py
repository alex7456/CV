import cv2
import numpy as np

def non_max_suppression(magnitude, angle):
    H, W = magnitude.shape
    output = np.zeros((H, W), dtype=np.float64)

    angle_deg = angle * 180.0 / np.pi
    angle_deg[angle_deg < 0] += 180

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            q = 255
            r = 255
            a = angle_deg[i, j]

            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif 22.5 <= a < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif 67.5 <= a < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif 112.5 <= a < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                output[i, j] = magnitude[i, j]
            else:
                output[i, j] = 0

    return output


def double_threshold(nms, low_ratio=0.1, high_ratio=0.3):
    high = nms.max() * high_ratio
    low = high * low_ratio

    strong = 255
    weak = 75

    H, W = nms.shape
    result = np.zeros((H, W), dtype=np.uint8)

    strong_i, strong_j = np.where(nms >= high)
    weak_i, weak_j = np.where((nms <= high) & (nms >= low))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return result


def process_image(path):
    # Считывание изображения
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Гауссово размытие
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Градиенты Собеля
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x)

    # Подавление немаксимумов
    nms = non_max_suppression(magnitude, angle)

    # Двойная пороговая фильтрация
    dt = double_threshold(nms)

    # Отображение
    nms_norm = cv2.normalize(nms, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imshow("Оригинал", gray)
    cv2.imshow("После NMS", nms_norm)
    cv2.imshow("Двойная пороговая фильтрация", dt)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


process_image("C:/Users/Alexandr/Desktop/cvision/4/img/2.png")
