import cv2

frame = cv2.imread("img/1.jpg", cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

cv2.namedWindow("Original picture", cv2.WINDOW_NORMAL)
cv2.imshow("Original picture", frame)

cv2.namedWindow("Changed picture", cv2.WINDOW_NORMAL)
cv2.imshow("Changed picture", hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()