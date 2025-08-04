import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))
from ultralytics import YOLO
from strongsort import StrongSORT  # StrongSORT 라이브러리 임포트
from pathlib import Path
import torch

# YOLOv8 모델 로드
model = YOLO("./runs/detect/train/weights/best.pt")  # 또는 사용자 정의 모델 경로

# StrongSORT 초기화

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_weights_path = Path('./osnet_x0_25_msmt17.pt')

tracker = StrongSORT(model_weights=model_weights_path, device=device, fp16=False)

# 비디오 캡처
cap = cv2.VideoCapture("./YOLO_V8/video_aquascape_2.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    if not ret:
        break

    # YOLOv8을 사용하여 객체 감지
    detections = []

    results = model(frame)
    result = results[0]

    if result.boxes is not None and result.boxes.data.numel() > 0:
        for box in result.boxes.data.cpu().numpy():  # box는 이미 np.ndarray
            x1, y1, x2, y2, conf, cls = box
            detections.append([x1, y1, x2, y2, conf, int(cls)])
    else:
        detections = np.empty((0, 6))  # 빈 detection은 2차원 배열로!

    # detections가 list면 numpy 배열로 변환
    if isinstance(detections, list):
        detections = np.array(detections)

    # numpy array detections에서 클래스 컬럼(마지막) 제거
    detections_no_class = detections[:, :5]

    # numpy → torch tensor, device 맞춰서 변환
    detections_tensor = torch.from_numpy(detections).float().to(device)

    # 업데이트 호출
    tracks = tracker.update(detections_tensor, ori_img=frame)

    # 추적 결과 시각화
    for track in tracks:
        x1, y1, x2, y2, track_id = track[:5]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
