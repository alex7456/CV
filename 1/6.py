import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
height, width = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if height is None:
        height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    cv2.line(frame, (center_x - 50, center_y - 200), (center_x + 50, center_y - 200), (0, 0, 255), 5)
    cv2.line(frame, (center_x - 200, center_y - 30), (center_x + 200, center_y - 30), (0, 0, 255), 5)
    cv2.line(frame, (center_x - 200, center_y - 30), (center_x - 200, center_y + 30), (0, 0, 255), 5)
    cv2.line(frame, (center_x + 200, center_y - 30), (center_x + 200, center_y + 30), (0, 0, 255), 5)
    cv2.line(frame, (center_x - 50, center_y + 200), (center_x + 50, center_y + 200), (0, 0, 255), 5)
    cv2.line(frame, (center_x - 200, center_y + 30), (center_x + 200, center_y + 30), (0, 0, 255), 5)
    cv2.line(frame, (center_x - 50, center_y + 30), (center_x - 50, center_y + 200), (0, 0, 255), 5)
    cv2.line(frame, (center_x + 50, center_y + 30), (center_x + 50, center_y + 200), (0, 0, 255), 5)
    cv2.line(frame, (center_x - 50, center_y - 30), (center_x - 50, center_y - 200), (0, 0, 255), 5)
    cv2.line(frame, (center_x + 50, center_y - 30), (center_x + 50, center_y - 200), (0, 0, 255), 5)

    cv2.imshow('1', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
