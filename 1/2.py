import cv2

v1 = cv2.imread("img/1.jpg", cv2.IMREAD_COLOR)
cv2.namedWindow("OutputPanel", cv2.WINDOW_NORMAL)
cv2.imshow("OutputPanel", v1)
cv2.waitKey(0)
cv2.destroyAllWindows()

v2 = cv2.imread("img/2.png", cv2.IMREAD_GRAYSCALE)
cv2.namedWindow("OutputPanel", cv2.WINDOW_FULLSCREEN)
cv2.imshow("OutputPanel", v2)
cv2.waitKey(0)
cv2.destroyAllWindows()

v3 = cv2.imread("img/3.bmp", cv2.IMREAD_ANYDEPTH)
cv2.namedWindow("OutputPanel", cv2.WINDOW_KEEPRATIO)
cv2.imshow("OutputPanel", v3)
cv2.waitKey(0)
cv2.destroyAllWindows()
