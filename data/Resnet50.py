import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ✅ 한글 폰트 설정 (한글 깨짐 방지)
plt.rcParams['font.family'] = "Malgun Gothic"

# ✅ 데이터 변환 설정
# 🔹 훈련 데이터 (Data Augmentation 적용)
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),  
    transforms.RandomRotation(20),  
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 🔹 테스트 데이터 (Augmentation 제거)
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  
])

# ✅ 데이터셋 로드
train_dataset = datasets.ImageFolder(root=r"C:\Users\AMI-DEEP3\Desktop\younguk\data\train", transform=train_transform)
test_dataset = datasets.ImageFolder(root=r"C:\Users\AMI-DEEP3\Desktop\younguk\data\test", transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ✅ 데이터 개수 확인
print(f"✅ Train 데이터 개수: {len(train_dataset)}")
print(f"✅ Test 데이터 개수: {len(test_dataset)}")

# ✅ ResNet50 모델 정의 (BatchNorm + Dropout 추가)
model = models.resnet50(pretrained=True)
num_classes = 2
model.fc = nn.Sequential(
    nn.Dropout(0.5),  
    nn.BatchNorm1d(model.fc.in_features),  
    nn.Linear(model.fc.in_features, num_classes)
)

# ✅ 학습 및 평가 루프
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ 손실 함수 및 옵티마이저 설정 (L2 Regularization 포함)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# ✅ 손실 저장 리스트
loss_history = []

# ✅ 학습 진행 (Epoch 100 + 조기 종료)
num_epochs = 100
early_stopping_patience = 10  
best_loss = float("inf")
early_stop_count = 0

print("\n🔹 학습 시작")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    loss_history.append(epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # 학습률 감소 적용
    scheduler.step(epoch_loss)

    # 조기 종료 조건 체크
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        early_stop_count = 0  
    else:
        early_stop_count += 1
        if early_stop_count >= early_stopping_patience:
            print(f"🚀 조기 종료: {epoch+1} Epoch에서 학습 중단!")
            break

# ✅ 손실 그래프 시각화
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-', color='b', label='Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss (ResNet50 + BatchNorm + Dropout)")
plt.legend()
plt.show()

# ✅ 테스트 평가
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []
test_images = []

# ✅ 정상 / 비정상 각각 10개씩 저장
normal_count, abnormal_count = 0, 0

with torch.no_grad():
    for images, labels in test_loader:  # 🔹 test_loader만 사용!
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # ✅ 정상 / 비정상 각각 10개씩 저장
        for i in range(len(images)):
            label = labels[i].cpu().numpy()
            if (label == 0 and normal_count < 10) or (label == 1 and abnormal_count < 10):
                test_images.append((images[i].cpu().numpy(), label, predicted[i].cpu().numpy()))
                if label == 0:
                    normal_count += 1
                else:
                    abnormal_count += 1

# ✅ 테스트 정확도 출력
accuracy = (correct / total) * 100
print(f"✅ ResNet50 (Dropout) 테스트 정확도: {accuracy:.2f}%")

# ✅ Confusion Matrix 시각화
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["정상", "비정상"], yticklabels=["정상", "비정상"])
plt.xlabel("예측값")
plt.ylabel("실제값")
plt.title("ResNet50 (Dropout) - Confusion Matrix")
plt.show()

# ✅ 정규화 해제 함수 (이미지를 원래 값으로 복원)
def denormalize(img):
    img = img * 0.5 + 0.5  
    img = np.clip(img, 0, 1)  
    return img

# ✅ 테스트 이미지 결과 시각화 (20개씩 페이지네이션)
class_names = ["정상", "비정상"]
images_per_page = 20  
num_pages = math.ceil(len(test_images) / images_per_page)

for page in range(num_pages):
    start_idx = page * images_per_page
    end_idx = min(start_idx + images_per_page, len(test_images))

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))  
    fig.suptitle(f"테스트 이미지 결과 (페이지 {page + 1}/{num_pages})", fontsize=16)

    for i, ax in enumerate(axes.flat):
        img_idx = start_idx + i
        if img_idx < len(test_images):
            image, true_label, pred_label = test_images[img_idx]
            image = denormalize(np.transpose(image, (1, 2, 0)))  

            ax.imshow(image)
            ax.set_title(f"정답: {class_names[true_label]}\n예측: {class_names[pred_label]}", fontsize=12,
                         color="green" if true_label == pred_label else "red")
            ax.axis("off")
        else:
            ax.axis("off")  

    plt.show()
