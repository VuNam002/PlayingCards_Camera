import cv2
import torch
import numpy as np
import os

# CẤU HÌNH: Đường dẫn đến file model best.pt
# Lưu ý: Bạn phải copy file best.pt vào đúng vị trí này
model_path = r'runs/detect/train/weights/best.pt'

if not os.path.exists(model_path):
    print(f"❌ LỖI: Không tìm thấy file model tại: {model_path}")
    print("👉 Vui lòng copy file 'best.pt' từ thư mục train cũ ra đường dẫn trên.")
    exit()

# Load model từ source 'local' (thư mục yolov5 trong dự án)
print(f"Đang load model từ: {model_path}...")
model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')

# Mở Webcam
cap = cv2.VideoCapture(0)
print("Đang mở webcam... Nhấn 'q' để thoát.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Nhận diện và hiển thị
    results = model(frame)
    cv2.imshow('YOLOv5 Webcam Predict', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()