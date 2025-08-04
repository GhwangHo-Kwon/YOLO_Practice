import cv2
from ultralytics import YOLO

class VideoLabeling:
    def __init__(self, video_path, model_path):
        self.video_path = video_path
        self.model = YOLO(model_path)  # YOLOv8 모델 로드
        self.output_dir = "labeled_frames/"
    
    def extract_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        # 영상에서 프레임을 추출
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = int(frame_count / fps)
            frame_filename = f"{self.output_dir}frame_{timestamp}.jpg"
            cv2.imwrite(frame_filename, frame)
            self.label_frame(frame_filename, frame)
            frame_count += 1
        
        cap.release()

    def label_frame(self, frame_filename, frame):
        results = self.model(frame)  # YOLOv8 모델을 사용한 객체 탐지
        labels = results.pandas().xywh[0]  # 라벨 데이터 추출 (bounding box, 클래스)

        # 라벨링 결과 출력 및 이미지에 bounding box 그리기
        for _, label in labels.iterrows():
            x1, y1, x2, y2 = int(label['xmin']), int(label['ymin']), int(label['xmax']), int(label['ymax'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label['name'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 라벨링 된 이미지 저장
        labeled_frame_filename = frame_filename.replace('.jpg', '_labeled.jpg')
        cv2.imwrite(labeled_frame_filename, frame)
