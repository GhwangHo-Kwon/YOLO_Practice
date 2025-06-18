import cv2
import argparse
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker
from ultralytics import YOLO

# 영상 및 모델 경로
video_path = './YOLO_V8/video_aquascape_2.mp4'
model = YOLO('./runs/detect/train/weights/best.pt')
model.to('cuda')

# 비디오 캡처 열기
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# 첫 프레임을 읽어 회전하고, 정확한 크기 측정
ret, frame = cap.read()
if not ret:
    raise RuntimeError("영상을 읽을 수 없습니다.")
frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
height, width = frame.shape[:2]

# VideoWriter 초기화
out = cv2.VideoWriter('output_detected_with_tracking.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (width, height))

# ByteTrack 초기화
tracker_args = {
    'track_thresh': 0.5,
    'match_thresh': 0.8,     # 조금 더 엄격하게
    'track_buffer': 30,      # 유지 시간 늘림
    'frame_rate': fps,       # 실제 FPS로 맞춤
    'min_box_area': 100,
    'iou_thresh': 0.3,       # 너무 낮지 않게
    'mot20': False,
    'max_time_lost': 60      # 객체 추적 유지 최대 프레임 수
}
tracker_args = argparse.Namespace(**tracker_args)
tracker = BYTETracker(tracker_args, frame_rate=tracker_args.frame_rate)

# 객체 추적 결과를 저장할 리스트
tracking_results = []

# 나머지 프레임 반복 처리
frame_count = 0  # 프레임 카운트 추가
retrack_counter = 0  # 재추적 횟수 추가

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame_count += 1  # 프레임 카운트 증가

    print(f"프레임 {frame_count} 처리 중...")

    # YOLO 객체 감지
    results = model.predict(source=frame, conf=0.25, iou=0.2, verbose=True)
    detections = results[0].boxes.data.cpu().numpy()

    if detections.size == 0:
        print(f"프레임 {frame_count} : 감지되지 않은 프레임입니다.")
        continue

    # 감지된 객체 로그
    print(f"프레임 {frame_count}: 감지된 객체 수: {len(detections)}")
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        print(f"객체 감지 - [x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, 신뢰도: {conf:.2f}, 클래스: {int(cls)}]")

    # ByteTrack을 사용하여 객체 추적
    detections_for_tracker = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if (x2 - x1) * (y2 - y1) < tracker_args.min_box_area:
            print(f"프레임 {frame_count}: 너무 작은 객체는 제외 - {det}")
            continue  # 너무 작은 객체는 제외
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        detections_for_tracker.append([center_x, center_y, width, height, conf])

    # detections_for_tracker를 numpy.ndarray로 변환
    detections_for_tracker = np.array(detections_for_tracker)

    if (detections_for_tracker.size == 0):
        print(f"프레임 {frame_count}: 감지된 객체 없음.")
        continue

    # 추적에 필요한 이미지 정보 (img_info 및 img_size)
    img_info = [height, width, 1.0]  # [height, width, aspect_ratio]
    img_size = (width, height)  # 이미지 크기 (width, height)

    # 추적된 객체
    print(f"detections_for_tracker shape: {detections_for_tracker.shape}")
    print(detections_for_tracker)
    print(f"img_info: {img_info}")

    online_tracks = tracker.update(detections_for_tracker, img_info, img_size)
    # 추적 실패시 YOLO로 다시 객체를 감지
    if online_tracks is None or len(online_tracks) == 0:  # 추적 객체가 없다면
        print(f"프레임 {frame_count}: 추적 실패, YOLO로 재감지")

        # 재감지 시도 제한
        tracker = BYTETracker(tracker_args, frame_rate=tracker_args.frame_rate)
        if retrack_counter < 3:  # 재감지를 최대 3번까지만 시도
            retrack_counter += 1
            # results = model.predict(source=frame, conf=0.25, iou=0.4, verbose=False)
            detections = results[0].boxes.data.cpu().numpy()

            detections_for_tracker = []
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if (x2 - x1) * (y2 - y1) < tracker_args.min_box_area:
                    print(f"프레임 {frame_count}: 너무 작은 객체는 제외 - {det}")
                    continue  # 너무 작은 객체는 제외

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                detections_for_tracker.append([center_x, center_y, width, height, conf])

            # detections_for_tracker를 numpy.ndarray로 변환
            detections_for_tracker = np.array(detections_for_tracker)

            if detections_for_tracker.size == 0:
                print(f"프레임 {frame_count}: 감지된 객체 없음.")
                continue

            # 다시 ByteTrack으로 추적
            online_tracks = tracker.update(detections_for_tracker, img_info, img_size)
        else:
            print(f"프레임 {frame_count}: 추적 실패, 재감지 제한.")
            retrack_counter = 0  # 재감지 횟수 리셋

    # 추적된 객체 결과를 JSON 형식으로 저장
    for track in online_tracks:
        track_id = track.track_id  # 객체 ID
        # track._tlwh가 [x1, y1, width, height] 형식일 경우
        x1, y1, width, height = track._tlwh  # 객체의 좌표 정보 가져오기
        x2, y2 = x1 + width, y1 + height  # x2, y2 계산 (오른쪽 하단 좌표)
        tracking_results.append({
            "track_id": track_id,
            "bbox": [x1, y1, x2, y2],
            "confidence": track.score
        })
        
        # 감지된 객체에 대한 라벨링
        # cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # YOLO 클래스 라벨 추가
        # cls_name = model.names[int(cls)]  # 클래스 이름 (YOLO 모델에서 제공)
        # label_with_cls = f'{cls_name}: {conf:.2f}'  # 클래스 이름과 신뢰도 표시
        # cv2.putText(frame, label_with_cls, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # 결과를 비디오에 저장
    out.write(frame)

    # 추적된 객체를 화면에 표시
    cv2.imshow("Detected and Tracked", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# 추적된 결과를 JSON 파일로 저장
import json
# tracking_results에 있는 값들이 numpy.float32 타입일 때 이를 float로 변환
def convert_to_float(obj):
    if isinstance(obj, np.float32):
        return float(obj)  # numpy.float32를 Python float으로 변환
    raise TypeError(f"Object of type {obj.__class__.__name__} is not serializable")

# tracking_results를 JSON으로 저장할 때
with open('tracking_results.json', 'w') as f:
    json.dump(tracking_results, f, indent=4, default=convert_to_float)