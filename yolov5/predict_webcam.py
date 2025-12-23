from ultralytics import YOLO
import os

# Đường dẫn tới model đã train của bạn
# Lưu ý: Nếu lệnh train trước đó bị lỗi, file này có thể chưa tồn tại.
model_path = r"runs\detect\train\weights\best.pt"

# Kiểm tra nếu không có model train thì dùng model mặc định để test
if not os.path.exists(model_path):
    print(f"Warning: Khong tim thay file model tai: {model_path}")
    print("Dang chuyen sang dung model mac dinh 'yolov8n.pt' de test camera...")
    model_path = "yolov8n.pt"

print(f"Dang chay model: {model_path} tren Webcam (source=0)...")

# Khởi tạo model và chạy dự đoán
model = YOLO(model_path)
model.predict(source="0", show=True)
