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
import matplotlib.font_manager as fm
import math

# ✅ 한글 폰트 설정 (한글 깨짐 방지)
plt.rcParams['font.family'] = fm.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name()

# ✅ 데이터 로드
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 정규화 적용
])

train_dataset = datasets.ImageFolder(root=r"C:\Users\AMI-DEEP3\Desktop\younguk\data\train", transform=transform)
test_dataset = datasets.ImageFolder(root=r"C:\Users\AMI-DEEP3\Desktop\younguk\data\test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ✅ 모델 정의 (ResNet18)
model = models.resnet18(pretrained=True)
num_classes = 2  # 정상 vs 비정상 이진 분류
model.fc = nn.Linear(model.fc.in_features, num_classes)

# ✅ 학습 및 평가 루프
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 손실 저장 리스트
loss_history = []

# ✅ 학습 진행 (Epoch 100)
num_epochs = 100
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

# ✅ 손실 그래프 시각화
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), loss_history, marker='o', linestyle='-', color='b', label='Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.show()

# ✅ 테스트 평가
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []
test_images = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 모든 테스트 이미지 저장
        for i in range(len(images)):
            test_images.append((images[i].cpu().numpy(), labels[i].cpu().numpy(), predicted[i].cpu().numpy()))

accuracy = (correct / total) * 100
print(f"✅ ResNet18 테스트 정확도: {accuracy:.2f}%")

# ✅ Confusion Matrix 시각화
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["정상", "비정상"], yticklabels=["정상", "비정상"])
plt.xlabel("예측값")
plt.ylabel("실제값")
plt.title("ResNet18 - Confusion Matrix")
plt.show()

# ✅ 정규화 해제 함수 (이미지를 원래 값으로 복원)
def denormalize(img):
    img = img * 0.5 + 0.5  # 정규화 반대로 변환
    img = np.clip(img, 0, 1)  # imshow()를 위한 범위 조정
    return img

# ✅ 모든 테스트 이미지 예측 결과 시각화 (20개씩 페이지네이션)
class_names = ["정상", "비정상"]
images_per_page = 20  # 한 페이지에 표시할 이미지 수l
num_pages = math.ceil(len(test_images) / images_per_page)  # 총 페이지 수

for page in range(num_pages):
    start_idx = page * images_per_page
    end_idx = min(start_idx + images_per_page, len(test_images))

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))  # 4행 5열 (20개씩 표시)
    fig.suptitle(f"테스트 이미지 결과 (페이지 {page + 1}/{num_pages})", fontsize=16)

    for i, ax in enumerate(axes.flat):
        img_idx = start_idx + i
        if img_idx < len(test_images):
            image, true_label, pred_label = test_images[img_idx]
            image = denormalize(np.transpose(image, (1, 2, 0)))  # (C, H, W) → (H, W, C)

            ax.imshow(image)
            ax.set_title(f"정답: {class_names[true_label]}\n예측: {class_names[pred_label]}", fontsize=12,
                         color="green" if true_label == pred_label else "red")
            ax.axis("off")
        else:
            ax.axis("off")  # 남은 빈칸 숨기기

    plt.show()