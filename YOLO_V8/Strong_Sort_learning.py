from strongsort.strong_sort import StrongSORT
import cv2
from ultralytics import YOLO
import torch
from pathlib import Path

# YOLO 모델 로드
model_weight = '../runs/detect/train/weights/best.pt'
model = YOLO(model_weight)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fp16 = False

model_weights_path = Path('./osnet_x0_25_msmt17.pt')

# Strong Sort 초기화
tracker = StrongSORT(model_weights=model_weights_path, device=device, fp16=fp16)

# 비디오 또는 카메라 입력
cap = cv2.VideoCapture('../YOLO_V8/video_aquascape_2.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # YOLO를 사용하여 물고기 탐지
    results = model(frame)  # YOLO 모델로 물고기 탐지
    boxes = results.xywh[0][:, :-1].cpu().numpy()  # bounding box 정보
    confidences = results.xywh[0][:, -1].cpu().numpy()  # confidence 정보

    # YOLO의 출력을 Strong Sort로 전달하여 추적
    outputs = tracker.update(boxes)  # 물고기 추적

    # 추적된 물고기 ID를 출력 (예: ID 표시, 경로 표시 등)
    for output in outputs:
        # output 예시: [x1, y1, x2, y2, track_id]
        x1, y1, x2, y2, track_id = output
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 출력
    cv2.imshow("Fish Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
