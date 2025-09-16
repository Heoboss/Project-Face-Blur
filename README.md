# 👤 초상권 보호를 위한 AI 얼굴 인식 블러 처리 시스템
**AI Face Recognition Blurring System for Portrait Rights Protection**

<br>

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

<br>

### 프로젝트 실행 결과
> 💡 **Tip**: 팀원을 제외한 일반인은 Blur 처리된 모습을 확인할 수 있습니다.

<img width="638" height="356" alt="결과사진1" src="https://github.com/user-attachments/assets/84505f96-77b1-4bb2-9e6e-2e20cb2619c7" />
<img width="639" height="352" alt="결과사진2" src="https://github.com/user-attachments/assets/17dfb41b-4975-44aa-be90-1207c3d264de" />


<br>

## 📜 목차
1. [**프로젝트 소개**](#1-프로젝트-소개)
2. [**기술 스택**](#2-기술-스택)
3. [**프로젝트 구조**](#3-프로젝트-구조)
4. [**설치 및 실행 방법**](#4-설치-및-실행-방법)
5. [**핵심 기능 분석**](#5-핵심-기능-분석)
6. [**프로젝트를 통해 배운 점**](#6-프로젝트를-통해-배운-점)

<br>

## 1. 프로젝트 소개

### 프로젝트 목표
본 프로젝트는 영상 속에서 사전 등록된 특정 인물(`Registered User`)과 그 외 불특정 다수(`Unknown`)를 AI 모델을 통해 분류하는 것을 목표로 합니다. 최종적으로, **사전에 등록된 사용자의 얼굴만 선택적으로 블러(Blur) 처리**하여 영상 콘텐츠에서의 프라이버시를 관리하는 시스템을 구축합니다.

### 주요 기능
- **얼굴 감지**: `MTCNN`을 활용하여 영상의 모든 프레임에서 실시간으로 인물의 얼굴 영역을 정확하게 감지합니다.
- **얼굴 분류**: 사전 학습된 ResNet 기반의 커스텀 `MyNet` 모델을 통해 감지된 얼굴이 '등록된 사용자'인지 '미등록 사용자'인지 분류합니다.
- **선택적 블러링**: 분류 결과에 따라 '등록된 사용자'로 판단된 얼굴 영역에만 `OpenCV`를 이용해 가우시안 블러 효과를 적용합니다.
- **데이터 자동화**: 입력된 원본 인물 사진들로부터 모델 학습에 필요한 얼굴 데이터셋을 자동으로 생성하고 분리합니다.

<br>

## 2. 기술 스택

| Category | Tech |
|---|---|
| **Language** | `Python 3.9` |
| **Framework / Library** | `PyTorch`, `OpenCV`, `facenet-pytorch`, `Pillow`, `NumPy` |
| **Environment** | `Anaconda` |
| **IDE** | `Visual Studio Code` |

<br>

## 3. 프로젝트 구조

* 📁 **face-blurring-project/**
    * 📁 **input_original_images/** - (사용자 입력) 원본 인물 사진 저장
        * 📁 **person_A/** - 등록할 사용자(블러 처리 대상) 사진
        * 📁 **unknown/** - 미등록 사용자(블러 처리 안 할 대상) 사진
    * 📁 **video_files/** - (사용자 입력) 원본 영상 및 결과 영상 저장
        * 📄 `example.mp4`
        * 📄 `output_blurred.mp4` - (자동 생성) 결과물
    * 📁 **custom_dataset/** - (자동 생성) 가공된 얼굴 데이터셋
    * 📁 **output_cropped_faces/** - (자동 생성) 원본 사진에서 잘라낸 얼굴 이미지
    * 📁 **saved_models/** - (자동 생성) 학습 완료된 모델 가중치 파일
    * 📄 `1_create_dataset.py` - 1. 원본 사진에서 얼굴 추출
    * 📄 `2_split_dataset.py` - 2. 데이터셋을 train/test로 분리
    * 📄 `3_train_model.py` - 3. AI 모델 학습
    * 📄 `4_run_blurring.py` - 4. 영상에 블러링 적용
    * 📄 `README.md` - 프로젝트 설명서

<br>

## 4. 설치 및 실행 방법

### 사전 준비
- `Anaconda`가 설치되어 있어야 합니다.

### 설치
**1. 저장소 복제**
   ```bash
   git clone https://github.com/Heoboss/Project-Face-Blur.git
   cd Project-Face-Blur
   ```
**2. Conda 가상 환경 생성 및 활성화**
   ```bash
   conda create --name faceblur_env python=3.9
   conda activate faceblur_env
   ```
**3. 필수 라이브러리 설치**
   ```bash
   # PyTorch (CPU 버전)
    conda install pytorch torchvision torchaudio -c pytorch
    # 기타 라이브러리
    conda install -c conda-forge opencv numpy pillow tqdm jupyterlab
    pip install facenet-pytorch
   ```
### 실행
**1. 데이터 준비**
  * `input_original_images/person_A/` 폴더에 등록할 사용자(블러 처리하지 않을 대상)의 얼굴 사진을 20장 이상 넣습니다.
  * `input_original_images/unknown/` 폴더에 그 외 다양한 인물들의 사진을 20장 이상 넣습니다.
  * `video_files/` 폴더에 테스트할 `원본 동영상(example.mp4)`을 넣습니다.

**2. 스크립트 순차 실행**  
  *VS Code 터미널에서 아래 명령어를 순서대로 실행합니다.*
```bash
python 1_create_dataset.py
python 2_split_dataset.py
python 3_train_model.py
python 4_run_blurring.py
```

**3. 결과 확인**
  * 모든 과정이 끝나면 `video_files/` 폴더에 `output_blurred.mp4` 파일이 생성됩니다.

<br>

## 5. 핵심 기능 분석
**1. 얼굴 분류 모델 (`3_train_model.py`)**
  * ResNet 아키텍처를 기반으로 한 커스텀 모델 `MyNet`을 구현하여 사용했습니다.

  * 소규모 데이터셋에서의 과적합(Overfitting)을 방지하고, 등록된 사용자와 미등록 사용자의 특징적 차이를 학습하는 데 중점을 두었습니다.

  * Windows 환경의 `multiprocessing` 오류를 해결하기 위해, 모델 학습 실행 코드를 `if __name__ == '__main__':` 블록 내에 배치하여 안정성을 확보했습니다.

**2. 선택적 블러링 로직 (`4_run_blurring.py`)**
  * `MTCNN`으로 영상 프레임 내 모든 얼굴을 탐지한 후, 각 얼굴 이미지를 학습된 `MyNet` 모델에 전달하여 추론을 수행합니다.

  * 모델이 얼굴을 `'unknown'`로 분류할 경우에만 해당 얼굴의 좌표를 `blur_targets` 리스트에 추가합니다.

  * 모든 얼굴에 대한 분류가 끝난 후, `blur_targets` 리스트에 포함된 좌표 영역에만 `cv2.GaussianBlur`를 적용하여 최종 결과물을 생성합니다.
```bash
# 'person_A' 클래스로 분류되면 블러 대상에 추가
if predicted_class == 'person_A':
    blur_targets.append(box)
```

<br>

## 6. 프로젝트를 통해 배운 점
  * 과적합 문제 해결: 초기에 단일 클래스(`person_A`)의 매우 적은 데이터로 학습했을 때, Loss가 0으로 수렴하고 정확도가 100%가 되는 심각한 과적합 현상을 경험했습니다. 이를 해결하기 위해 '모르는 사람'에 해당하는 `unknown` 네거티브 클래스를 추가하고 데이터 양을 늘림으로써, 모델이 단순 암기가 아닌 분류를 위한 특징을 학습하도록 유도할 수 있었습니다.
  * 환경 설정의 중요성: Windows 환경에서 PyTorch의 `DataLoader` 멀티프로세싱 사용 시 발생하는 `RuntimeError`를 해결하기 위해 `if __name__ == '__main__':`의 중요성을 체감했습니다. 또한, VS Code 터미널에서 Conda를 사용하기 위한 `conda init`과 PowerShell 실행 정책 설정 등, 안정적인 개발 환경 구축의 필요성을 배웠습니다.
  * 컴퓨터 비전 파이프라인 이해: 원본 이미지 수집부터 데이터 전처리(얼굴 추출), 모델 학습, 추론, 그리고 후처리(블러링)에 이르는 컴퓨터 비전 프로젝트의 전체적인 파이프라인을 직접 구축하고 디버깅하며 실무적인 경험을 쌓을 수 있었습니다.
