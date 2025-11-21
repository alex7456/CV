import cv2
import numpy as np

image = cv2.imread('img/2.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



kernel = np.ones((3,3), np.uint8)


opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

opening_then_closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Original', gray)

cv2.imshow('Opening (removes salt)', opening)
cv2.imshow('Closing (removes pepper)', closing)
cv2.imshow('Opening then Closing', opening_then_closing)


cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imwrite('opening.jpg', opening)
cv2.imwrite('closing.jpg', closing)