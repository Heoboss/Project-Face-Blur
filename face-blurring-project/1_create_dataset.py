# 1_create_dataset.py
import cv2
import os
import torch
from facenet_pytorch import MTCNN
from PIL import Image

# 입력 및 출력 디렉토리 설정
image_directory = 'input_original_images'
output_directory = 'output_cropped_faces'

# 출력 디렉토리 생성
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# MTCNN 모델 초기화 (얼굴 감지)
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

print("얼굴 데이터셋 생성을 시작합니다...")

# 각 인물 폴더에 대해 반복
for person_name in os.listdir(image_directory):
    person_input_dir = os.path.join(image_directory, person_name)
    person_output_dir = os.path.join(output_directory, person_name)

    if not os.path.isdir(person_input_dir):
        continue

    if not os.path.exists(person_output_dir):
        os.makedirs(person_output_dir)

    # 폴더 내의 각 이미지 파일에 대해 반복
    file_count = 0
    for filename in os.listdir(person_input_dir):
        image_path = os.path.join(person_input_dir, filename)
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"이미지 파일을 열 수 없습니다: {image_path}, 오류: {e}")
            continue
            
        # 얼굴 감지
        boxes, _ = mtcnn.detect(img)

        if boxes is not None:
            for i, box in enumerate(boxes):
                # 얼굴 영역 자르기
                face = img.crop(box)
                
                # 자른 얼굴 이미지 저장
                output_filename = f"{person_name}_{file_count}_{i}.jpg"
                output_path = os.path.join(person_output_dir, output_filename)
                face.save(output_path)
            file_count += 1
            print(f"[{person_name}] {filename}에서 얼굴 감지 및 저장 완료.")
        else:
            print(f"[{person_name}] {filename}에서 얼굴을 감지하지 못했습니다.")

print("모든 이미지 처리가 완료되었습니다.")