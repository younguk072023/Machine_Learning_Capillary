import os

# 이미지가 저장된 부모 폴더 경로
dataset_path = r"C:\Users\AMI-DEEP3\Desktop\younguk\data" 

# 이미지 확장자 목록
valid_extensions = ('.jpg', '.jpeg', '.png')

# 폴더별 이미지 개수 확인 함수
def count_images_in_folder(folder_path):
    return len([f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)])

# 정상 (0) 폴더 개수 확인
normal_folder = os.path.join(dataset_path, "0")
normal_count = count_images_in_folder(normal_folder)

# 비정상 (1) 폴더 개수 확인
abnormal_folder = os.path.join(dataset_path, "1")
abnormal_count = count_images_in_folder(abnormal_folder)

print(f"✅ 정상(U자) 데이터 개수: {normal_count}장")
print(f"✅ 비정상(비U자) 데이터 개수: {abnormal_count}장")
print(f"📌 총 이미지 개수: {normal_count + abnormal_count}장")
