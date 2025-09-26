import cv2

cap = cv2.VideoCapture("img/video.mp4", cv2.CAP_ANY)

mode = 0 

window_width = 640
window_height = 480


cv2.namedWindow('Video - Multiple Formats', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video - Multiple Formats', window_width, window_height)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    

    if mode == 0:
        display_frame = frame
        mode_text = "ORIGINAL (BGR)"
    elif mode == 1:
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
        mode_text = "GRAYSCALE"
    elif mode == 2:
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mode_text = "HSV"
    elif mode == 3:
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        mode_text = "LAB"
    
    display_frame = cv2.resize(display_frame, (window_width, window_height))
    
   
    
    cv2.imshow('Video - Multiple Formats', display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27: 
        break
    elif key == ord('1'):
        mode = 0
    elif key == ord('2'):
        mode = 1
    elif key == ord('3'):
        mode = 2
    elif key == ord('4'):
        mode = 3

cap.release()
cv2.destroyAllWindows()