import cv2
import numpy as np


cap = cv2.VideoCapture(0)


kernel = np.ones((5, 5), np.uint8)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

  
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

 
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)


    opened = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

  
    cv2.imshow("Original", frame)
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Opened", opened)
    cv2.imshow("Closed", closed)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
