import json
import cv2
import torch
from ultralytics import YOLO

# 경로 설정
video_path = './YOLO_V8/Sample_data/robot_fish.mp4'
model = YOLO('./runs/detect/train4/weights/best.pt')  # 학습된 Guppy 탐지 모델

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
out = cv2.VideoWriter('tracked_robot_fish.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (width, height))

# 추적된 객체 정보 저장할 리스트 초기화
tracked_objects = []

# 🔁 영상 전체 추적 실행
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 회전
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 🔍 객체 추적
    results = model.track(source=frame, conf=0.6, persist=True, verbose=False, save_txt=True, tracker="./YOLO_V8/bytetrack.yaml")

    # 결과 시각화
    annotated_frame = results[0].plot()
    out.write(annotated_frame)

    # 추적된 객체 정보 수집
    frame_data = []
    for result in results[0].boxes:
        # 각 객체의 ID, bbox, confidence 값을 추출
        obj_id = result.id if result.id is not None else -1  # 객체 ID가 None이면 -1로 처리
        bbox = result.xywh[0].cpu().numpy().tolist()  # 바운딩 박스 (x, y, w, h)
        conf = result.conf[0].cpu().item()  # 신뢰도 (Tensor에서 값을 추출)

        # 객체 정보 딕셔너리 생성
        frame_data.append({
            'id': obj_id,
            'bbox': bbox,
            'confidence': conf
        })

    # 이 프레임의 객체 정보 저장
    tracked_objects.append(frame_data)

    # 출력
    cv2.imshow("Tracked", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 추적된 객체 데이터를 JSON 형식으로 저장
# Tensor 값을 JSON 직렬화 가능하도록 변환 후 저장
with open('tracked_robot_fish.json', 'w') as json_file:
    # tracked_objects의 모든 Tensor 값을 숫자 값으로 변환
    # (예: bbox, confidence 등)
    def convert_tensor(obj):
        if isinstance(obj, (float, int)):
            return obj
        elif isinstance(obj, list):
            return [convert_tensor(i) for i in obj]
        elif isinstance(obj, dict):
            return {key: convert_tensor(value) for key, value in obj.items()}
        elif isinstance(obj, torch.Tensor):
            return obj.item()  # Tensor에서 값을 추출
        return obj

    json.dump([convert_tensor(frame) for frame in tracked_objects], json_file, indent=4)

cap.release()
out.release()
cv2.destroyAllWindows()
