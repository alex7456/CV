import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    b, g, r = frame[cy, cx]

    if r >= g and r >= b:
        color = (0, 0, 255)
    elif g >= r and g >= b:
        color = (0, 255, 0)
    else:
        color = (255, 0, 0)

    cv2.rectangle(frame, (cx - 50, cy - 200), (cx + 50, cy + 200), color, -1)
    cv2.rectangle(frame, (cx - 200, cy - 50), (cx + 200, cy + 50), color, -1)

    cv2.imshow("1", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

