import cv2

cap = cv2.VideoCapture(0)



while True:
    
    ret, frame = cap.read()
    if not ret:
        break

  
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

   
    cv2.imshow("Original", frame)
    cv2.imshow("HSV", hsv_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
