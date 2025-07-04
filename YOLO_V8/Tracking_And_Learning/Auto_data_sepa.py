import os
import random
import shutil

# ✅ 설정
base_image_dir = './YOLO_V8/Sample_data/base_data/images/'   # 전체 이미지 폴더 경로
base_label_dir = './YOLO_V8/Sample_data/base_data/labels/'   # 전체 라벨 폴더 경로

output_base = './YOLO_V8/Sample_data/'  # 출력 디렉토리

# 분할 비율 설정 (총합 1.0)
train_ratio = 0.7
valid_ratio = 0.15
test_ratio = 0.15

# 지원 이미지 확장자
image_exts = ['.jpg', '.jpeg', '.png']

# 디렉토리 생성 함수
def make_dirs(base):
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(base, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base, split, 'labels'), exist_ok=True)

# 이미지 파일 불러오기
all_images = [f for f in os.listdir(base_image_dir) if os.path.splitext(f)[1].lower() in image_exts]
random.shuffle(all_images)

# 분할
total = len(all_images)
num_train = int(total * train_ratio)
num_valid = int(total * valid_ratio)

train_files = all_images[:num_train]
valid_files = all_images[num_train:num_train + num_valid]
test_files = all_images[num_train + num_valid:]

# 디렉토리 생성
make_dirs(output_base)

# 복사 함수
def copy_files(file_list, split_name):
    for img_file in file_list:
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + '.txt'

        img_src = os.path.join(base_image_dir, img_file)
        label_src = os.path.join(base_label_dir, label_file)

        img_dst = os.path.join(output_base, split_name, 'images', img_file)
        label_dst = os.path.join(output_base, split_name, 'labels', label_file)

        shutil.copy2(img_src, img_dst)
        if os.path.exists(label_src):
            shutil.copy2(label_src, label_dst)

# 복사 실행
copy_files(train_files, 'train')
copy_files(valid_files, 'valid')
copy_files(test_files, 'test')

print(f"✅ 데이터 분할 완료!")
print(f"총 이미지: {total}")
print(f"Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")