Dưới đây là một ví dụ file `README.md` phù hợp với mô tả của bạn:

```markdown
# Vietnamese-Laotian Machine Translation

## 📌 Mô tả dự án

Dự án tạo ra mô hình dịch máy giữa tiếng Việt và tiếng Lào

## 👥 Thành viên nhóm


## 🧠 Mô hình sử dụng

- **M2M100**: Một mô hình dịch đa ngôn ngữ không cần phụ thuộc vào tiếng Anh làm ngôn ngữ trung gian. Mô hình có thể dịch trực tiếp giữa hơn 100 ngôn ngữ, bao gồm cả tiếng Việt và tiếng Lào.

## 📂 Các tệp chính trong dự án

```
.
├── m2m100_vilo.ipynb   # File Jupiter Notebook dùng để việc xử lý dữ liệu, tạo, train và evaluate model
├── bleu.py             # Script tính BLEU Score từ kết quả được ghi ở file
├── results/            # Chứa một số kết quả của việc chạy test trên mô hình đã được train
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
  - Việt -> Lào:  BLEU score ~56 
