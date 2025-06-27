from ultralytics import YOLO

class YOLOTrainer:
    def __init__(self, train_data_path, val_data_path, model_type="yolov8"):
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.model = YOLO(model_type)  # YOLOv8 모델 로드

    def train(self, epochs=50, batch_size=16):
        # 데이터셋 경로와 하이퍼파라미터 설정
        train_dataset = f"{self.train_data_path}/train"
        val_dataset = f"{self.val_data_path}/val"
        
        # 학습 시작
        self.model.train(
            data={"train": train_dataset, "val": val_dataset},
            epochs=epochs,
            batch_size=batch_size
        )
    
    def evaluate(self):
        # 모델 평가
        results = self.model.val()
        print(f"Validation Results: {results}")
