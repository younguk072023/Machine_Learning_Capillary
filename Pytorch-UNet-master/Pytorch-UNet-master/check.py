import os
import cv2
import numpy as np

# 마스크 파일이 있는 폴더 경로
mask_dir = "C:\\Users\\AMI-DEEP3\\Desktop\\final_data\\train\\mask"

# 마스크 파일 목록 가져오기
mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.jpg')]

# 첫 번째 마스크 이미지 로드
if len(mask_files) > 0:
    sample_mask_path = os.path.join(mask_dir, mask_files[0])
    mask_img = cv2.imread(sample_mask_path, cv2.IMREAD_GRAYSCALE)  # 흑백으로 로드

    # 1️⃣ 마스크 이미지의 기본 정보 출력
    print(f"✅ 마스크 이미지 로드: {sample_mask_path}")
    print(f"🔹 이미지 크기: {mask_img.shape}")
    print(f"🔹 데이터 타입: {mask_img.dtype}")

    # 2️⃣ 픽셀 값의 최소/최대 범위 확인
    min_val = np.min(mask_img)
    max_val = np.max(mask_img)
    unique_vals = np.unique(mask_img)

    print(f"🔹 최소값: {min_val}, 최대값: {max_val}")
    print(f"🔹 고유값 샘플: {unique_vals[:10]}")

    # 3️⃣ 마스크가 0과 255로만 이루어진 이진 마스크인지 확인
    if set(unique_vals).issubset({0, 255}):
        print("✅ 마스크는 이진(binary) 마스크입니다. (0과 255)")
    else:
        print("⚠️ 마스크가 다중 클래스 값을 가질 가능성이 있습니다.")
else:
    print("❌ 마스크 파일이 없습니다!")
