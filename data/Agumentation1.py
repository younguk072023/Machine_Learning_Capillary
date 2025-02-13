import torchvision.transforms as transforms
from PIL import Image
import os

# 데이터 증강 설정
augmentation = transforms.Compose([
    transforms.RandomRotation(10),  # 최대 ±10도 회전
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # 밝기 및 대비 조절
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5)),  # 블러 추가
    transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),  # 10% 내외로 크기 조정
])

# 원본 비정상 이미지 폴더
original_folder = r"C:\Users\AMI-DEEP3\Desktop\younguk\data\train\1"  # 비정상 데이터 폴더

# 증강된 이미지 저장 폴더
augmented_folder = r"C:\Users\AMI-DEEP3\Desktop\younguk\data\train\1"
os.makedirs(augmented_folder, exist_ok=True)

# 데이터 증강 적용 및 저장
for img_name in os.listdir(original_folder):
    img_path = os.path.join(original_folder, img_name)
    image = Image.open(img_path)

    for i in range(5):  # 한 장당 5개의 증강 이미지 생성
        augmented_image = augmentation(image)
        save_path = os.path.join(augmented_folder, f"aug_{i}_{img_name}")
        augmented_image.save(save_path)

print("✅ 비정상 데이터 증강 완료!")
