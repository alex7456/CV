import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640) 
cap.set(4, 480)

fourcc = cv2.VideoWriter_fourcc(*'XVID')  
video_writer = cv2.VideoWriter('img/output.mov', fourcc, 20.0, (640, 480))  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    video_writer.write(frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
