import cv2
import time
import os

def run_tracker(video_path, method_id):
    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    _, frame = cap.read()

    cv2.startWindowThread()
    bbox = cv2.selectROI("ROI", frame, False)
    cv2.destroyWindow("ROI")
    time.sleep(0.3)

    if method_id == 1:
        tracker = cv2.legacy.TrackerCSRT_create()
        name = "CSRT"
    elif method_id == 2:
        tracker = cv2.legacy.TrackerKCF_create()
        name = "KCF"
    elif method_id == 3:
        tracker = cv2.legacy.TrackerMOSSE_create()
        name = "MOSSE"
    else:
        raise ValueError("Неверный метод (1-CSRT, 2-KCF, 3-MOSSE)")

    tracker.init(frame, bbox)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"output/tracked_{name}_{os.path.basename(video_path)}",
                          fourcc, fps_video, (width, height))

    frame_count, lost_frames = 0, 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        ok, box = tracker.update(frame)
        frame_count += 1
        if not ok:
            lost_frames += 1
        else:
            x, y, w, h = map(int, box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        out.write(frame)
        cv2.imshow(f"Tracking – {name}", frame)
        if cv2.waitKey(20) & 0xFF == 27:
            break

    end_time = time.time()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    elapsed = end_time - start_time
    processing_fps = frame_count / elapsed if elapsed > 0 else 0
    fps_diff = processing_fps - fps_video
    lost_ratio = (lost_frames / frame_count) * 100 if frame_count > 0 else 0
    tracking_stability = 100 - lost_ratio

    print(f"\nМетод: {name}")
    print(f"Разница FPS (трекинг – видео): {fps_diff:+.2f}")
    print(f"Потери рамки: {lost_frames} ({lost_ratio:.2f} %)")
    print(f"Стабильность трекинга: {tracking_stability:.2f} %")
    print(f"FPS трекинга: {processing_fps:.2f}")
    print(f"FPS видео: {fps_video:.2f}")

# Пример запусков
# run_tracker("video/bear.mp4", 3)
# run_tracker("video/camels.mp4", 2)
# run_tracker("video/gorilla.mp4", 3)
# run_tracker("video/lions.mp4", 2)
run_tracker("video/tiger.mp4", 1)