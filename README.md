# Vietnamese-Laotian Machine Translation

## 📌 Mô tả dự án

Dự án tạo ra mô hình dịch máy giữa tiếng Việt và tiếng Lào

## 👥 Thành viên nhóm
Lương Mạnh Linh     22021215
Lê Quang Thắng      22021209
Đặng Thanh Quang    22021134

## 🧠 Mô hình sử dụng

- **M2M100**: Một mô hình dịch đa ngôn ngữ không cần phụ thuộc vào tiếng Anh làm ngôn ngữ trung gian. Mô hình có thể dịch trực tiếp giữa hơn 100 ngôn ngữ, bao gồm cả tiếng Việt và tiếng Lào.
-**Transformer tiêu chuẩn**: Mô hình tự tạo thích hợp cho tác vụ dịch máy
## 📂 Các tệp chính trong dự án

```
.
├── report/             # Thư mục chứa báo cáo
├── slide/              # Thư mục chứa slide trình bày
├── src/                # Thư mục chứa mã nguồn
│   ├── m2m100_vilo.ipynb   # Notebook xử lý dữ liệu, train và evaluate model
│   └── bleu.py             # Script tính BLEU Score từ kết quả
├── results/            # Thư mục chứa kết quả trên tập test

```

## 🚀 Cách chạy

### 1. Train và evaluate mô hình trong `m2m100_vilo.ipynb`
    - Điều chỉnh đường dẫn đến thư mục chứa data
    - Sau khi chạy evaluate, sẽ sinh ra 2 tệp tương ứng `label.txt` và `pred.txt`

### 2. Đánh giá mô hình sử dụng `bleu.py`
    - Điều chỉnh đường dẫn đến thư mục chứa file `label` và `pred`
    - Chạy main

## 📊 Đánh giá
- Sử dụng BLEU score để đánh giá chất lượng bản dịch.
- Kết quả trên tập test:  
  - Việt -> Lào:  BLEU Score ~56 (sacrebleu) và ~29 (evaluate)
  - Lào -> Việt: BLEU Score ~41 (cả sacrebleu và evaluate)

## Sử dụng mô hình
- Model M2M100 209M của nhóm đã có trên [HuggingFace](https://huggingface.co/luongmanhlinh)
- Transformer tiêu chuẩn có thể được tải xuống tại [đây](https://drive.google.com/drive/folders/1_SPVJjF4urbIOiTJ1woGjxnHQS1Gu-V4)
