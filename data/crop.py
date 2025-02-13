import cv2
import os

# 원본 이미지 폴더 (이미지가 저장된 위치)
source_folder = r"C:\Users\AMI-DEEP3\Desktop\younguk\data\d"

# 저장할 폴더 경로
save_folder_normal = r"C:\Users\AMI-DEEP3\Desktop\younguk\data\0"
save_folder_abnormal = r"C:\Users\AMI-DEEP3\Desktop\younguk\data\1"

# 폴더 생성
os.makedirs(save_folder_normal, exist_ok=True)
os.makedirs(save_folder_abnormal, exist_ok=True)

# 이미지 파일 목록 불러오기
image_files = [f for f in os.listdir(source_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# ROI의 크기 (128x128) 고정
ROI_SIZE = 128
selected_x, selected_y = 0, 0  # 초기 ROI 위치

def mouse_callback(event, x, y, flags, param):
    """ 마우스 움직임에 따라 ROI의 위치를 변경하는 콜백 함수 """
    global selected_x, selected_y
    if event == cv2.EVENT_MOUSEMOVE:
        selected_x, selected_y = x, y  # 마우스 위치 업데이트

for img_name in image_files:
    img_path = os.path.join(source_folder, img_name)
    
    # 이미지 불러오기 (BGR 모드 유지)
    image = cv2.imread(img_path)

    # 초기 ROI 위치 설정
    selected_x, selected_y = image.shape[1] // 2, image.shape[0] // 2

    # 창 생성 및 마우스 이벤트 등록
    cv2.namedWindow("Select Blood Vessel")
    cv2.setMouseCallback("Select Blood Vessel", mouse_callback)

    while True:
        display_img = image.copy()
        
        # ROI 영역 표시 (128x128 크기 고정)
        cv2.rectangle(display_img, (selected_x, selected_y), 
                      (selected_x + ROI_SIZE, selected_y + ROI_SIZE), (0, 255, 0), 2)

        cv2.imshow("Select Blood Vessel", display_img)

        key = cv2.waitKey(1) & 0xFF

        # Enter 키(🔲) 누르면 ROI 확정 & 저장 (계속 같은 이미지에서 진행)
        if key == 13:
            cropped = image[selected_y:selected_y+ROI_SIZE, selected_x:selected_x+ROI_SIZE]

            # **BGR -> RGB 변환 후 저장** (PIL 사용 가능)
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

            # 사용자에게 저장할 라벨 선택 (정상/비정상)
            while True:
                label = input(f"{img_name}: 정상(U자) → '0', 비정상(비U자) → '1' 입력: ").strip().lower()
                if label in ['n', 'a']:
                    break
                print("잘못된 입력입니다. 'n' 또는 'a'를 입력하세요.")

            # 파일 저장 경로 설정
            save_filename = f"vessel_{img_name}_x{selected_x}_y{selected_y}.jpg"
            save_path = os.path.join(save_folder_normal if label == 'n' else save_folder_abnormal, save_filename)

            # 이미지 저장
            cv2.imwrite(save_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))  # 다시 BGR로 변환하여 저장
            print(f"✅ 저장 완료: {save_path}")

        # ESC 키(❌) 누르면 다음 이미지로 이동
        if key == 27:
            print(f"🔄 {img_name} - 다음 이미지로 이동합니다.")
            cv2.destroyAllWindows()
            break

    # 창 닫기
    cv2.destroyAllWindows()

print("✅ 모든 이미지 처리 완료!")
