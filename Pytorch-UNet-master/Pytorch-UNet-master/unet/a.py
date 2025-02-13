from unet_model import UNet  # UNet 모델 불러오기
import torch
# 모델 생성
model = UNet(n_channels=1, n_classes=1)

# 더미 입력 데이터 생성
dummy_input = torch.randn(1, 1, 480, 640)  # (batch, channels, height, width)

# 모델 실행
output = model(dummy_input)

# 출력 크기 확인
print(output.shape)  # Expected: torch.Size([1, 1, 480, 640])
