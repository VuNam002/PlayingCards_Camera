# Hệ thống Nhận diện Lá bài thời gian thực

Dự án này sử dụng mô hình YOLOv8 để nhận diện các lá bài trong video trực tiếp từ webcam.

## Demo Ứng dụng

Dưới đây là hình ảnh minh họa kết quả nhận diện của ứng dụng trong thực tế. Hệ thống có khả năng xác định nhiều lá bài khác nhau trong một khung hình.

![Demo nhận diện lá bài](demo/Screenshot%202025-12-23%20200546.png)
![Demo nhận diện lá bài](demo/Screenshot%202025-12-23%20202050.png)
![Demo nhận diện lá bài](demo/Screenshot%202025-12-23%20202138.png)

## Cơ sở Lý thuyết và Ứng dụng trong Mô hình YOLOv8

Trong khi dự án sử dụng trực tiếp mô hình YOLOv8 đã được huấn luyện để thực hiện nhận diện, sức mạnh, tốc độ và sự hiệu quả của mô hình này không phải là ngẫu nhiên. Chúng bắt nguồn từ các nguyên lý tối ưu hóa toán học sâu sắc. Phần này sẽ đi sâu vào cơ sở lý thuyết đó, giải thích cách **Thuật toán Subgradient** được ứng dụng để giải quyết các bài toán như **Lasso** và **Phân cụm lồi**, qua đó tạo ra các mô hình học sâu nhỏ gọn và hiệu suất cao như YOLOv8.

### 1. Nền tảng: Thuật toán Subgradient

Thuật toán Subgradient là công cụ nền tảng để giải quyết các bài toán tối ưu hóa lồi khi hàm mục tiêu không khả vi tại một số điểm. Đây là tình huống cực kỳ phổ biến trong học máy hiện đại, đặc biệt là khi sử dụng các hàm chính quy hóa (regularization) như L1-norm.

Phương pháp lặp của thuật toán có dạng: `x_{k+1} = x_k - α_k * g_k`, nơi `g_k` là một subgradient của hàm mục tiêu. Ý tưởng này cho phép chúng ta tìm điểm tối ưu ngay cả khi không có đạo hàm rõ ràng.

### 2. Lasso, Tối ưu hóa YOLOv8 và Thuật toán Subgradient

**Liên kết trực tiếp đến dự án:** Tốc độ real-time của ứng dụng này phụ thuộc hoàn toàn vào sự nhỏ gọn và hiệu quả của mô hình `yolov8n.pt`. Một trong những kỹ thuật quan trọng để tạo ra các mô hình như vậy là **"model pruning"** (tỉa mô hình), và nguyên lý đằng sau nó chính là **Lasso (L1 Regularization)**.

- **Bài toán:** Làm thế nào để giảm kích thước của một mạng neural network (như YOLOv8) mà không làm giảm đáng kể độ chính xác?
- **Giải pháp với Lasso:** Trong quá trình huấn luyện hoặc tinh chỉnh (fine-tuning) mô hình, một thành phần chính quy hóa L1 (`λ * ||w||₁`) được thêm vào hàm mất mát. `w` ở đây là các trọng số của mạng. Thành phần này "phạt" các trọng số có giá trị lớn, và quan trọng hơn, nó **ép buộc các trọng số không quan trọng phải bằng 0**.
- **Kết quả:** Quá trình này tạo ra một mô hình "thưa" (sparse), nơi nhiều kết nối thần kinh đã bị loại bỏ (trọng số bằng 0). Các kênh (channels) hoặc thậm chí toàn bộ các lớp (layers) có thể được xác định là không cần thiết và bị "tỉa" đi, tạo ra một mô hình nhỏ hơn, nhanh hơn, lý tưởng cho các ứng dụng thời gian thực như dự án này.
- **Vai trò của Subgradient:** Vì hàm chính quy hóa L1 không khả vi tại 0, bài toán tối ưu hóa này không thể được giải bằng Gradient Descent thông thường. Thay vào đó, các thuật toán thuộc họ Subgradient (ví dụ: Proximal Gradient Descent, FISTA) được sử dụng để tìm ra lời giải thưa một cách hiệu quả.

**Kết luận:** Như vậy, có một mối liên hệ trực tiếp: **Thuật toán Subgradient** cho phép giải bài toán **Lasso**, và nguyên lý của Lasso được áp dụng trong các kỹ thuật **tỉa mô hình** để tạo ra các phiên bản YOLOv8 hiệu quả (`yolov8n.pt`) mà dự án này đang sử dụng.

### 3. Phân cụm lồi và Phân tích Dữ liệu trong Học máy

**Liên kết trực tiếp đến dự án:** Trước khi huấn luyện một mô hình nhận diện mạnh mẽ, việc hiểu rõ cấu trúc của bộ dữ liệu là cực kỳ quan trọng. Đây là giai đoạn "Phân tích Dữ liệu Khám phá" (Exploratory Data Analysis), và Phân cụm lồi là một công cụ mạnh mẽ cho việc này.

- **Bài toán:** Trong bộ dữ liệu hình ảnh các lá bài, có những nhóm lá bài nào tự nhiên giống nhau về mặt hình ảnh (ví dụ: các lá 'K', 'Q', 'J' có thể có các đặc trưng khuôn mặt tương đồng, hoặc các lá bài cùng một nước (cơ, rô, chuồn, bích) có thể có cấu trúc tương tự)? Việc xác định các nhóm này giúp dự đoán những trường hợp nào mô hình sẽ dễ bị nhầm lẫn.
- **Giải pháp với Phân cụm lồi:**
    1. Trích xuất vector đặc trưng (feature vectors) từ mỗi hình ảnh trong bộ dữ liệu huấn luyện bằng một mạng CNN.
    2. Áp dụng thuật toán Phân cụm lồi lên không gian các vector đặc trưng này.
    - **Kết quả:** Thuật toán sẽ tự động nhóm các hình ảnh vào các cụm lồi, cho thấy các "họ hình ảnh" (visual families) tồn tại trong dữ liệu. Ví dụ, kết quả có thể cho thấy tất cả các lá bài số '7' được nhóm lại gần nhau, nhưng lá '7 cơ' lại nằm ở rìa của cụm đó, gần với cụm của các lá bài '8 cơ' hơn, cho thấy một sự tương đồng về hình ảnh.
- **Vai trò của Subgradient:** Hàm mục tiêu của Phân cụm lồi chứa các thành phần không khả vi, do đó một lần nữa, Thuật toán Subgradient lại là phương pháp cần thiết để tìm ra các cụm một cách tối ưu.

**Kết luận:** Mặc dù không chạy trong script `predict_webcam.py` cuối cùng, Phân cụm lồi là một bước quan trọng trong **quy trình nghiên cứu và phát triển** một hệ thống thị giác máy tính. Nó cung cấp những hiểu biết sâu sắc về dữ liệu, giúp định hướng các chiến lược tăng cường dữ liệu (data augmentation) và thiết kế kiến trúc mô hình.

## Kết quả Thực nghiệm
Dưới đây là một số kết quả từ quá trình huấn luyện mô hình, cho thấy hiệu suất của mô hình nhận diện.

### 1. Ma trận nhầm lẫn (Confusion Matrix)
Ma trận nhầm lẫn cho thấy độ chính xác của mô hình trên từng lớp (mỗi lá bài). Các giá trị trên đường chéo chính càng cao, mô hình càng chính xác.

![Ma trận nhầm lẫn](weights/runs/detect/train7/confusion_matrix_normalized.png)

### 2. Đường cong PR (Precision-Recall Curve)
Đường cong PR thể hiện sự cân bằng giữa Precision (độ chính xác) và Recall (độ phủ) tại các ngưỡng tin cậy khác nhau. Một đường cong càng gần góc trên bên phải càng cho thấy hiệu suất tốt.

![Đường cong PR](weights/runs/detect/train7/BoxPR_curve.png)

### 3. Kết quả nhận diện trên tập huấn luyện
Hình ảnh dưới đây minh họa kết quả nhận diện của mô hình trên một lô (batch) dữ liệu huấn luyện. Các hộp bao và nhãn cho thấy mô hình đã học cách xác định vị trí và loại của các lá bài.

![Kết quả nhận diện](weights/runs/detect/train7/val_batch2_labels.jpg)


## Yêu cầu
- Python 3.8 trở lên
- Pip (trình quản lý gói cho Python)
- Webcam

## Hướng dẫn Cài đặt

1.  **Sao chép Repository**
    ```bash
    git clone <https://github.com/VuNam002/PlayingCards_Camera>
    cd PlayingCards_Camera
    ```

2.  **Tạo và Kích hoạt Môi trường ảo**

    Việc sử dụng môi trường ảo là một thông lệ tốt để quản lý các gói phụ thuộc của dự án.

    - Trên Windows:
      ```bash
      python -m venv venv
      .\venv\Scripts\activate
      ```
    - Trên macOS và Linux:
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```

3.  **Cài đặt các Gói phụ thuộc**

    Cài đặt tất cả các thư viện cần thiết được liệt kê trong file `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    *Lưu ý: Quá trình cài đặt `torch` có thể mất một chút thời gian.*

## Cấu hình

1.  **Model Nhận diện (`best.pt`)**
    - Script `predict_webcam.py` yêu cầu file model đã được huấn luyện có tên là `best.pt`.
    - Theo mặc định, script sẽ tìm model tại đường dẫn: `weights/runs/detect/train7/weights/best.pt`.
    - Hãy chắc chắn rằng bạn đã có file model này từ quá trình huấn luyện hoặc đã tải về và đặt nó vào đúng thư mục.
    - Nếu model của bạn nằm ở một vị trí khác, hãy cập nhật biến `model_path` trong file `predict_webcam.py`:
      ```python
      # CẤU HÌNH: Đường dẫn đến file model best.pt
      model_path = r'duong/dan/den/model/cua/ban/best.pt'
      ```

## Chạy Ứng dụng

Sau khi hoàn tất cài đặt và cấu hình, bạn có thể chạy ứng dụng bằng lệnh sau:

```bash
python predict_webcam.py
```

- Một cửa sổ hiển thị hình ảnh từ webcam sẽ xuất hiện ở chế độ toàn màn hình.
- Các lá bài được nhận diện sẽ được đánh dấu bằng các hộp chữ nhật và nhãn.
- Để thoát khỏi ứng dụng, hãy nhấn phím `q`.
