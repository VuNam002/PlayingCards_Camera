import cv2
from ultralytics import YOLO
import os

# CẤU HÌNH: Đường dẫn đến file model best.pt
model_path = r'weights/runs/detect/train7/weights/best.pt'

# Kiểm tra sự tồn tại của model
if not os.path.exists(model_path):
    print(f"LỖI: Không tìm thấy file model tại: {model_path}")
    print("Vui lòng kiểm tra lại đường dẫn và đảm bảo file 'best.pt' tồn tại.")
    exit()

# Load mô hình YOLOv8
print(f"Đang load model từ: {model_path}...")
model = YOLO(model_path)

# Mở Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" LỖI: Không thể mở webcam.")
    exit()

print("Đang mở webcam... Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Kết thúc stream hoặc lỗi đọc frame.")
        break
    
    # Chạy nhận diện trên frame
    results = model(frame, stream=True)

    # Xử lý và hiển thị kết quả
    for r in results:
        annotated_frame = r.plot()
        cv2.imshow("YOLOv8 Webcam Predict", annotated_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp
cap.release()
cv2.destroyAllWindows()
print("Đã đóng webcam.")