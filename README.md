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