# YOLO_Practice
YOLO v8 or v5 학습

## YOLO v8 설치

### 가상환경 설정
1. pyenv install 3.10.11
    - 설치
        ```shell
        >> pyenv install 3.10.11
        ```
    - 설치 후 python 3.10.11 사용 설정
        ```shell
        >> pyenv global 3.10.11
        ```
    - 확인
        ```shell
        >> pyenv versions
        ```

2. 가상환경 생성
    - 디렉토리 이동
        ```shell
        >> cd C:\Source\YOLO_Practice
        ```
    - 가상환경 생성
        ```shell
        >> python -m venv yolov_env
        ```
    - 가상환경 활성화
        ```shell
        >> yolov_env\Scripts\activate
        ```

3. 필수 패키지 설치
    - PyTorch + CUDA 11.8
        ```shell
        >> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
    - YOLOv8
        ```shell
        >> pip install ultralytics
        ```
    - (선택) YOLOv5를 위해 추가 패키지 설치
        ```shell
        >> pip install opencv-python matplotlib pandas seaborn
        ```

4. 설치확인 - [test.py](./YOLO_V8/test.py)

5. `필수!` : 깃허브 커밋 전 .gitignore에서 가상환경 하위 폴더 등록

## 데이터 학습

### 데이터 준비
1. 폴더 준비
    - datasets > images > train > *.jpg
    - datasets > images > val > *.jpg
    - datasets > labels > train > *.txt
    - datasets > labels > val > *.txt
    - *.txt 파일은 YOLO 라벨 포맷 (class_id center_x center_y width height), 좌표는 모두 0~1로 정규화
        - 예: 0 0.512 0.634 0.234 0.321

2. 데이터 준비
    1. 위 *.txt 파일의 좌표값은 사진 데이터에서 실제 객체가 있는 좌표를 뜻함
        - 즉 전체 화면에서 인식해야할 객체의 좌표를 사각형으로 그린 것
        - 좌표 값을 찾기 위해서 노가다 해야함
        - 하지만 라벨링 도구를 사용하면 그나마 쉬움 (노가다 해야하는건 변함없음...)
    2. 라벨링 도구
        - [LabelImg](https://github.com/HumanSignal/labelImg) : 로컬 라벨링 툴
            - Python GUI 라벨링 도구
            - 이미지를 하나씩 열고 마우스로 바운딩 박스를 그리면 *.txt 자동 생성됨
            - YOLO 포맷으로 저장 가능
        - [Roboflow](https://roboflow.com/) : 웹 기반 라벨링 툴
            - 웹에서 이미지 업로드 → 라벨링 → 자동 .txt 생성
            - 전처리(Augmentation) 기능까지 있음

3. data.yaml 파일 생성 - [data.yaml](./YOLO_V8/guppy_data.yaml)
    - data.yaml : 클래스와 이미지 경로를 정의
    - nc : 클래스 수 / names : 클래스 이름 리스트

4. 데이터 학습 - [learning.py](./YOLO_V8/learning.py)
    - epochs, batch, imgsz 값은 자유 조절
    - yolov8n.pt = Nano / yolov8s.pt = Small

    ```python
    # 모델 학습시 사전 학습한 모델 선택
    Final_model = YOLO('yolov8s.pt')  # yolov8s.pt 은 small 모델임

    # 학습 시작
    Result_Final_model = Final_model.train(
        data="guppy_data.yaml",  # 데이터셋 정의 파일 (train, val, names 등이 정의된 YAML 파일)
        epochs = 200,  # 총 학습 epoch 수 (200번 전체 데이터셋을 반복 학습)
        imgsz = 640,  # 입력 이미지 크기 (640x640)
        lr0 = 0.005,  # 초기 학습률 (기본값보다 살짝 높음)
        momentum = 0.9,  # SGD 모멘텀 값
        weight_decay = 0.005,  # 가중치 감소 (정규화, 과적합 방지)
        warmup_epochs = 5,  # 처음 5 epoch 동안 워밍업 (느리게 시작)
        warmup_momentum = 0.8,  # 워밍업 동안의 모멘텀
        warmup_bias_lr = 0.01,  # 바이어스 항 학습률 (워밍업용)
        batch = 32,  # 배치 크기 (GPU에 따라 조절)
        device = 0,  # 0번 GPU 사용 (단일 GPU일 경우 그대로 사용)

        mosaic = 1.0,  # Mosaic 증강 (4장의 이미지 합성) 사용
        mixup = 0.0,  # MixUp 비활성화 (이미지 두 장 합성은 사용 안 함)
        fliplr = 0.5,  # 좌우 반전 확률 50%
        flipud = 0.3,  # 상하 반전 확률 30%
        degrees = 15.0,  # 최대 ±15도 회전
        translate = 0.1,  # 10% 이내의 평행이동 허용
        scale = 0.3,  # 최대 ±30% 확대/축소
        shear = 2.0,  # 최대 ±2도 기울이기
        hsv_h = 0.015,  # 색상(Hue) 조절 범위
        hsv_s = 0.7,  # 채도(Saturation) 조절 범위
        hsv_v = 0.4,  # 명도(Value) 조절 범위

        cache = True,  # 데이터를 메모리에 미리 로딩해 학습 속도 향상
        verbose = False,  # 학습 중 출력 간소화
        rect = False,  # True로 하면 사각형 비율 유지 학습 (영상 비율 유지용)
        cos_lr = True,  # Cosine learning rate 스케줄링 사용 (초반 느리게, 후반 빠르게 감소)
        profile = True  # 학습 시작 시 모델 구조 및 속도 프로파일 출력
        )
    ```

5. 데이터 학습 시 추가 팁
    - 로봇 물고기처럼 형태가 고정된 객체라면 rotate, translate, scale만 살짝 쓰고 mixup은 끄는 게 좋음
    - yolov8s.pt는 YOLOv8의 Small 모델로, 속도와 정확도 균형이 잘 잡힌 모델
        - GPU 메모리 여유 있다면 yolov8m.pt 또는 yolov8l.pt로도 확장 가능
    - batch=32는 적당하지만, GPU 메모리 부족하면 CUDA out of memory 에러 날 수 있음
        - 그럴 경우 batch=16 이하로 줄이기
    - cache=True는 빠르긴 하지만 RAM을 많이 사용하므로 주의
    - YOLOv8에서 회전을 제어하는 파라미터 : degrees
        - 이미지와 객체를 최대 ±15도 회전시킴
        - 학습 중 무작위로 이미지가 -15° ~ +15° 사이로 회전됨
        - 회전된 이미지에 맞게 바운딩 박스도 함께 회전되어 조정됨
        - 사용하는 이유
            - 물체가 다양한 각도로 찍히는 경우 (예: 공중에서 본 차량, 비대칭 물체)
            - 물고기처럼 방향이 자주 바뀌는 객체
            - 데이터 양이 적을 때 모델에 다양한 시각적 상황을 인위적으로 제공
        - 주의사항
            - 너무 큰 degrees 값을 설정하면 바운딩 박스가 깨지거나 이미지 왜곡이 심해질 수 있음
            - 일반적으로 10~15도가 안정적인 범위
            - 만약 물체가 항상 정방향(예: CCTV 고정 각도)으로 나오는 경우라면, 회전 증강은 오히려 노이즈가 될 수 있음

### 수중 환경 최적화 방법
1. 이미지 전처리 (Preprocessing)
    - YOLO 입력 전에 이미지 품질을 개선하는 작업

    | 기법                              | 설명                            | 코드 예시                                                                              |
    | ------------------------------- | ----------------------------- | ---------------------------------------------------------------------------------- |
    | **CLAHE**                       | 대비 향상 (어두운 환경에 효과적)           | `cv2.createCLAHE()`                                                                |
    | **White Balance**               | 색상 왜곡 보정 (청록색 물 제거)           | 수동 RGB 스케일링                                                                        |
    | **Histogram Equalization**      | 전체 밝기/색상 분포 개선                | `cv2.equalizeHist()`                                                               |
    | **Dehazing / Color Correction** | 색상 복원 (OpenCV X, 전용 라이브러리 필요) | [https://github.com/zhanghang1989/DCPDN](https://github.com/zhanghang1989/DCPDN) 등 |
    | **Bilateral Filter**            | 노이즈 제거 + 경계 유지                | `cv2.bilateralFilter()`                                                            |
- 예시
    ```python
    def preprocess_image(img):
        # Convert to LAB for CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # White balance correction
        result = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2LAB)
        avg_a = result[..., 1].mean()
        avg_b = result[..., 2].mean()
        result[..., 1] = result[..., 1] - ((avg_a - 128) * (result[..., 0] / 255.0) * 1.1)
        result[..., 2] = result[..., 2] - ((avg_b - 128) * (result[..., 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

        return result
    ```

2. YOLO Threshold 조정
    - YOLO의 출력에서 confidence threshold와 NMS threshold를 조정해 정확도를 높일 수 있음
        
        ```python
        results = model(frame, conf=0.3, iou=0.4)[0]
        ```
        
    | 파라미터   | 설명                  | 추천값                                 |
    | ------ | ------------------- | ----------------------------------- |
    | `conf` | 탐지 신뢰도 필터링          | `0.25 ~ 0.4` (수중은 낮춰야 놓치는 객체 ↓)     |
    | `iou`  | NMS (겹치는 박스 제거 임계값) | `0.4 ~ 0.6` (너무 낮으면 겹치는 물고기 하나만 남음) |



### 학습결과 및 테스트
1. 결과 확인
    - 학습이 끝나면 runs/detect/train/ 폴더에 결과 저장
    - weights/best.pt : 최종 학습된 모델
    - results.png : 정확도, 손실 그래프
    - confusion_matrx.png : 클래스 별 인식 성능
    - predictions.jpg : 검출 시각화 결과

2. 학습한 모델로 테스트
    - 결과는 runs/detect/predict/ 폴더에 저장
    - 결과 안의 box 좌표도 results[0].boxes.xyxy 등으로 추출 가능

## 실시간 추적

### 탐지 모델

#### ByteTrack
1. 설치
    ```shell
    pip install lap cython
    pip install git+https://github.com/ifzhang/ByteTrack.git
    ```
2. ByteTrack 설정
    | 파라미터 이름        | 기본값 예시  | 설명                                                                      |
    | -------------- | ------- | ----------------------------------------------------------------------- |
    | `track_thresh` | `0.5`   | **추적 시작 임계값**. 탐지 신뢰도가 이 값 이상이어야 추적 시작. 너무 낮으면 오탐 추적, 너무 높으면 검출 누락.     |
    | `match_thresh` | `0.8`   | **기존 트랙과 새로운 탐지 간 매칭 허용 오차**. 1에 가까울수록 덜 까다롭게 매칭 (ID 유지 ↑), 너무 높으면 오매칭↑ |
    | `track_buffer` | `30`    | 추적 ID 유지 시간 (프레임 수). 이 시간 내에 다시 보이면 ID 유지, 지나면 삭제됨.                     |
    | `frame_rate`   | `30`    | FPS 값. 추적기 내부의 시간 기반 로직에서 사용됨. 영상이나 웹캠의 실제 FPS에 맞춰야 정확도 ↑               |
    | `mot20`        | `False` | MOT20 벤치마크용 설정. 일반적으로 `False`.                                          |
    | `min_box_area` | `0`     | 너무 작은 박스는 무시 (잡음 제거용). 작은 물고기 필터링에 사용 가능.                               |
    | `iou_thresh`   | `0.5`   | **IoU 기반의 추적 매칭 임계값**. 기존 추적과 새 탐지 간의 IoU가 이 이상이어야 매칭.                  |
    | `track_thresh` | `0.5`   | 다시 나오는 객체에 대해 추적 재시작을 허용하는 confidence 기준                                |
    | `use_byte`     | `True`  | ByteTrack 알고리즘 핵심 기능 사용 여부 (True 권장)                                    |
    | `match_thresh` | `0.8`   | 외형 정보 없이 IoU로만 매칭할 때의 임계값                                               |
    | `with_reid`    | `False` | Deep SORT처럼 appearance feature로도 추적할지 여부 (ByteTrack 기본은 False)          |

    ```python
    tracker_args = {
    "track_thresh": 0.4,       # 너무 낮으면 오탐, 너무 높으면 놓침
    "match_thresh": 0.7,       # 물고기 유사하게 생겼으면 너무 높이면 ID 바뀔 수 있음
    "track_buffer": 30,        # 30프레임(1초) 정도 ID 유지
    "frame_rate": 30,          # 웹캠이나 영상의 실제 FPS에 맞춤
    "min_box_area": 100,       # 너무 작은 노이즈 제거 (원하는 만큼 조절)
    "iou_thresh": 0.5,         # 트랙 매칭 기준
    "mot20": False             # 일반 환경이면 False
    }
    ```

#### Deep SORT

##### ReID란? (Re-Identification)
- **ReID (Re-identification, 재식별)**은 다음과 같은 작업을 의미
    - "이 프레임에서 본 객체 A가, 다음 프레임 또는 다른 카메라에서도 여전히 같은 객체 A인가?"
- 예시
    - 프레임 1: 물고기 A가 있음
    - 프레임 5: 비슷하게 생긴 물고기 B도 있음
        - 단순히 위치나 IoU로만 판단하면 ID가 바뀔 수 있음
        - 그래서 외형(appearance) 특징을 사용해서 객체를 "식별"하는 것이 ReID

##### ReID 학습이 필요한 이유
| 상황                 | 문제가 생기는 이유   |
| ------------------ | ------------ |
| 비슷한 물고기 여러 마리      | 위치만으로 구분 불가능 |
| 빠르게 움직여서 박스가 튐     | IoU로 연결 실패   |
| 물고기가 잠깐 가려졌다 다시 등장 | ID가 새로 부여됨   |

- 이럴 때 ReID는 물고기의 색상, 무늬, 형태 등을 벡터(특징값)로 추출해서 ID를 유지할 수 있게 도와줌

##### ReID 학습 방식 요약
1. 데이터 수집
    - 동일한 물고기를 여러 각도/시점에서 촬영
    - 각 물고기마다 ID를 붙인 데이터셋 구성 필요
2. 모델 학습 (ImageNet 기반 CNN 사용 가능)
    - 입력: 물고기 이미지
    - 출력: 특징 벡터 (예: 512차원 벡터)
    - 목적: 같은 ID는 비슷한 벡터, 다른 ID는 멀어지도록 학습
3. 추적기와 통합
    - 프레임마다 YOLO → 물고기 박스 추출
    - 박스 안 이미지를 ReID 모델에 넣어서 feature vector 추출
    - 벡터 유사도 + 위치로 추적 매칭 수행 (Deep SORT 방식)

##### 실제 ReID를 적용할 수 있는 경우
- 고정된 수조 안에서 개별 물고기를 구별하고 싶다
    - 연구, 개체 행동 분석
- 다른 카메라/프레임 간 물고기 ID 유지가 중요하다
- 외형 구분이 확실한 종 (무늬, 색상이 다른 물고기들)

##### 적용이 어렵거나 불필요한 경우
- 물고기 외형이 거의 동일함 (ReID로도 구분 불가능)
- 목표가 단순 "몇 마리 추적"만일 때
- 실시간 속도 최우선 (ReID는 연산량이 큼)