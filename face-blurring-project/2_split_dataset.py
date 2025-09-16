# 2_split_dataset.py
import os
import shutil
import random

# 경로 설정
source_directory = 'output_cropped_faces'
train_dir = './custom_dataset/train/'
test_dir = './custom_dataset/test/'
split_ratio = 0.9  # 90%를 훈련 데이터로 사용

# 디렉토리 생성
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

print("데이터셋 분리를 시작합니다...")

# 소스 디렉토리의 각 인물 폴더에 대해 작업
for person_name in os.listdir(source_directory):
    person_source_path = os.path.join(source_directory, person_name)
    if not os.path.isdir(person_source_path):
        continue

    # 각 인물별 train/test 폴더 생성
    person_train_path = os.path.join(train_dir, person_name)
    person_test_path = os.path.join(test_dir, person_name)
    os.makedirs(person_train_path, exist_ok=True)
    os.makedirs(person_test_path, exist_ok=True)

    # 파일 목록 가져오고 섞기
    files = os.listdir(person_source_path)
    random.shuffle(files)

    # 분리 지점 계산
    split_point = int(len(files) * split_ratio)
    
    # 파일 이동
    train_files = files[:split_point]
    test_files = files[split_point:]

    for file_name in train_files:
        shutil.move(os.path.join(person_source_path, file_name), os.path.join(person_train_path, file_name))
    print(f"[{person_name}] 훈련 데이터 {len(train_files)}개 이동 완료.")
    
    for file_name in test_files:
        shutil.move(os.path.join(person_source_path, file_name), os.path.join(person_test_path, file_name))
    print(f"[{person_name}] 테스트 데이터 {len(test_files)}개 이동 완료.")

print("데이터셋 분리가 완료되었습니다.")