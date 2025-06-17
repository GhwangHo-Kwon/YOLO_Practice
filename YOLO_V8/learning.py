from ultralytics import YOLO

# 모델 불러오기 (Nano 모델: 빠르고 가벼움)
model = YOLO('yolov8n.pt')  # 또는 yolov8s.pt

# 학습 시작
model.train(
    data='guppy_data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0  # GPU 사용, 없다면 'cpu'
)