import os
import random
import shutil

# Đường dẫn gốc của bạn
src_root = r"C:\Users\Admin\Downloads\PlayingCards"
# Thư mục mới cho dữ liệu nhỏ
dst_root = r"C:\Users\Admin\Downloads\PlayingCards_Small"

def create_subset(split, num_images):
    img_src = os.path.join(src_root, split, "images")
    lbl_src = os.path.join(src_root, split, "labels")
    
    img_dst = os.path.join(dst_root, split, "images")
    lbl_dst = os.path.join(dst_root, split, "labels")
    
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)

    all_images = [f for f in os.listdir(img_src) if f.endswith(('.jpg', '.png', '.jpeg'))]
    selected_images = random.sample(all_images, min(num_images, len(all_images)))

    for img_name in selected_images:
        # Copy ảnh
        shutil.copy(os.path.join(img_src, img_name), os.path.join(img_dst, img_name))
        # Copy nhãn tương ứng
        label_name = os.path.splitext(img_name)[0] + ".txt"
        if os.path.exists(os.path.join(lbl_src, label_name)):
            shutil.copy(os.path.join(lbl_src, label_name), os.path.join(lbl_dst, label_name))

# Thực hiện lọc
print("Đang trích xuất dữ liệu nhỏ...")
create_subset("train", 800)  # Lấy 800 ảnh train
create_subset("valid", 200)  # Lấy 200 ảnh valid
print(f"Hoàn thành! Dữ liệu mới tại: {dst_root}")