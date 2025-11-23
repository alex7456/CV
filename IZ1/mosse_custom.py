import numpy as np
import cv2

class MOSSE:
    def __init__(self, learning_rate=0.125):
        self.lr = learning_rate
        self.H = None
        self.G = None
        self.size = None

        self.x = self.y = self.w = self.h = 0  # текущий bbox

    def create_gaussian(self, w, h, sigma=2.0):
        x = np.arange(0, w)
        y = np.arange(0, h)
        xx, yy = np.meshgrid(x, y)
        cx, cy = w // 2, h // 2
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
        return g

    def preprocess(self, img):
        img = np.log(img + 1)
        img = (img - img.mean()) / (img.std() + 1e-5)
        return img

    def init(self, frame, bbox):
        self.x, self.y, self.w, self.h = map(int, bbox)

        patch = cv2.cvtColor(frame[self.y:self.y+self.h, self.x:self.x+self.w], cv2.COLOR_BGR2GRAY)
        patch = patch.astype(np.float32)

        patch = self.preprocess(patch)

        g = self.create_gaussian(self.w, self.h)
        self.G = np.fft.fft2(g)

        X = np.fft.fft2(patch)
        self.H = (self.G * np.conj(X)) / (X * np.conj(X) + 1e-5)

    def update(self, frame):

        H, W = frame.shape[:2]

        # ограничиваем координаты bbox рамками изображения
        self.x = max(0, min(self.x, W - self.w))
        self.y = max(0, min(self.y, H - self.h))

        # вырезаем область
        patch = frame[self.y:self.y+self.h, self.x:self.x+self.w]

        # если пустой — считаем потерю
        if patch.size == 0:
            return False, (self.x, self.y, self.w, self.h)

        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch = patch.astype(np.float32)
        patch = self.preprocess(patch)

        Z = np.fft.fft2(patch)
        R = self.H * np.conj(Z)
        r = np.fft.ifft2(R).real

        dy, dx = np.unravel_index(np.argmax(r), r.shape)

        self.x += dx - self.w // 2
        self.y += dy - self.h // 2

        # снова ограничиваем
        self.x = max(0, min(self.x, W - self.w))
        self.y = max(0, min(self.y, H - self.h))

        # обновление фильтра
        H_new = (self.G * np.conj(Z)) / (Z * np.conj(Z) + 1e-5)
        self.H = (1 - self.lr) * self.H + self.lr * H_new

        return True, (self.x, self.y, self.w, self.h)