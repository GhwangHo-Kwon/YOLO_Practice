from ultralytics import YOLO
import cv2

# 경로 설정
video_path = './YOLO_V8/video_aquascape_2.mp4'
model = YOLO('./runs/detect/train/weights/best.pt')  # 학습된 Guppy 탐지 모델

# 비디오 읽기
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# 첫 프레임 확인 및 크기 설정
ret, frame = cap.read()
if not ret:
    raise RuntimeError("영상을 읽을 수 없습니다.")

frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
height, width = frame.shape[:2]

# 비디오 저장 객체 초기화
out = cv2.VideoWriter('output_tracked.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (width, height))

# 🔁 영상 전체 추적 실행
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 회전
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 🔍 객체 추적
    results = model.track(source=frame, conf=0.1, persist=True, verbose=False, tracker="./YOLO_V8/bytetrack.yaml")

    # 결과 시각화
    annotated_frame = results[0].plot()
    out.write(annotated_frame)

    # 출력
    cv2.imshow("Tracked", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
