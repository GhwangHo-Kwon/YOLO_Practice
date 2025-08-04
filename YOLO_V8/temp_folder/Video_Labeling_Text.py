import cv2
import os
from ultralytics import YOLO

class VideoLabeling:
    def __init__(self, video_path, model_path):
        self.video_path = video_path
        self.model = YOLO(model_path)  # YOLOv8 모델 로드
        self.output_dir = "labeled_frames/"
        self.create_output_dir()

    def create_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

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
            self.label_frame(frame_filename, frame, timestamp)
            frame_count += 1
        
        cap.release()

    def label_frame(self, frame_filename, frame, timestamp):
        results = self.model(frame)  # YOLOv8 모델을 사용한 객체 탐지
        labels = results.pandas().xywh[0]  # 라벨 데이터 추출 (bounding box, 클래스)

        # 텍스트 라벨 파일 경로
        label_txt_filename = frame_filename.replace('.jpg', '.txt')

        # 라벨링 결과 출력 및 이미지에 bounding box 그리기
        with open(label_txt_filename, 'w') as f:
            for _, label in labels.iterrows():
                # YOLO 형식에 맞게 좌표 변환 (xywh: 상대적 중심 좌표와 크기)
                x_center = (label['xmin'] + label['xmax']) / 2 / frame.shape[1]
                y_center = (label['ymin'] + label['ymax']) / 2 / frame.shape[0]
                width = (label['xmax'] - label['xmin']) / frame.shape[1]
                height = (label['ymax'] - label['ymin']) / frame.shape[0]

                # 클래스 ID는 현재 'name'에 매핑된 숫자를 사용
                class_id = label['class']

                # 텍스트 파일에 라벨 쓰기 (YOLO 형식)
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

                # 라벨링 된 이미지에 bounding box 그리기
                x1, y1, x2, y2 = int(label['xmin']), int(label['ymin']), int(label['xmax']), int(label['ymax'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label['name'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 라벨링 된 이미지 저장
        labeled_frame_filename = frame_filename.replace('.jpg', '_labeled.jpg')
        cv2.imwrite(labeled_frame_filename, frame)
