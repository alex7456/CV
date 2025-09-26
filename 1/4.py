import cv2

def copy_video():
    cap = cv2.VideoCapture("img/video.mp4")

    ret, frame = cap.read()
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("img/output.mp4", fourcc, 25, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("video_window", frame)
        out.write(frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

copy_video()
