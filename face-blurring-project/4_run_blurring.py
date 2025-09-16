# 4_run_blurring.py
import cv2
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import os

# 3_train_model.py 에서 정의한 MyNet 모델 구조를 그대로 가져옵니다.
#from_3_train_model import MyNet, Block 

# ▼▼▼ 여기에 3_train_model.py에서 복사한 클래스 코드를 붙여넣으세요 ▼▼▼
class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class MyNet(nn.Module):
    def __init__(self, block, layers, num_classes=3):
        super(MyNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
# ▲▲▲ 여기까지 붙여넣기 ▲▲▲

# --- 설정 ---
VIDEO_PATH = 'video_files/example.mp4'
MODEL_PATH = 'saved_models/face_classifier_model.pth'
OUTPUT_PATH = 'video_files/output_blurred.mp4'

# 클래스 이름은 custom_dataset/train 폴더 기준으로 자동 설정
CLASS_NAMES = sorted(os.listdir('custom_dataset/train'))
NUM_CLASSES = len(CLASS_NAMES)

# 얼굴이 특정인일 확률(logit)이 이 값보다 크면 블러 처리하지 않음
# 모델과 데이터에 따라 실험적으로 조절해야 합니다.
CONFIDENCE_THRESHOLD = 2.5 

# --- 모델 및 장치 초기화 ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 얼굴 탐지기
mtcnn = MTCNN(keep_all=True, device=device)

# 얼굴 분류기
model = MyNet(Block, [2, 2, 2, 2], num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 비디오 처리 ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"오류: 비디오 파일을 열 수 없습니다. 경로: {VIDEO_PATH}")
    exit()

# 비디오 속성
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print("비디오 블러링 처리를 시작합니다...")
with tqdm(total=total_frames, desc="Processing Frames") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> RGB 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # 얼굴 감지
        boxes, _ = mtcnn.detect(pil_img)

        blur_targets = []
        if boxes is not None:
            for box in boxes:
                # 얼굴 영역 자르기
                face = pil_img.crop(box)
                
                # 분류를 위한 이미지 전처리
                img_tensor = transform(face).unsqueeze(0).to(device)

                # 얼굴 분류
                with torch.no_grad():
                    outputs = model(img_tensor)
                    max_logit, preds = torch.max(outputs, 1)
                    predicted_class = CLASS_NAMES[preds.item()]
                    
                    print(f"감지된 얼굴: {predicted_class}, Logit: {max_logit.item():.2f}")
                    
                    # 신뢰도가 낮으면(일반인이면) 블러 대상에 추가
                    # if max_logit.item() < CONFIDENCE_THRESHOLD:
                    #     blur_targets.append(box)
                    # 'unknown' 클래스로 분류되면 블러 대상에 추가
                    if predicted_class == 'unknown':
                        blur_targets.append(box)

        # 블러 처리
        frame_h, frame_w, _ = frame.shape
        for box in blur_targets:
            x1, y1, x2, y2 = [int(b) for b in box]

            # ---▼▼▼ 안전장치 코드 추가 ▼▼▼---
            # 좌표가 프레임 경계를 벗어나지 않도록 보정합니다.
            y1 = max(0, y1)
            x1 = max(0, x1)
            y2 = min(frame_h - 1, y2)
            x2 = min(frame_w - 1, x2)
            # ------------------------------------

            # 보정된 좌표로도 영역이 유효한지(넓이가 있는지) 한번 더 확인합니다.
            if x1 < x2 and y1 < y2:
                face_roi = frame[y1:y2, x1:x2]

                # 블러 적용
                blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
                frame[y1:y2, x1:x2] = blurred_face

        # 처리된 프레임 저장
        out.write(frame)
        pbar.update(1)

        # (선택사항) 실시간으로 보기
        # cv2.imshow('Face Blurring', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\n처리 완료! 결과 비디오가 {OUTPUT_PATH}에 저장되었습니다.")