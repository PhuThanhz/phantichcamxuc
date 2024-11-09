import pandas as pd
import joblib
from pyvi import ViTokenizer
from hdfs import InsecureClient
import re
import string


# Kết nối đến HDFS
client = InsecureClient("http://localhost:50070")  # Thay đổi URL HDFS nếu cần

# Đọc mô hình đã lưu
MODEL_PATH = "sentiment_model.pkl"
clf = joblib.load(MODEL_PATH)


# Hàm chuẩn hóa văn bản
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = ViTokenizer.tokenize(text)  # Tokenize văn bản tiếng Việt
    return text


# Đọc tệp .crash (dữ liệu đầu vào)
def read_crash_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    data = []
    for i in range(0, len(lines), 2):  # Đọc theo cặp (ID + nhận xét)
        text = lines[i + 1].strip().replace('"', "")  # Xóa dấu nháy kép nếu có
        data.append(text)
    return data


# Đọc tệp dữ liệu .crash
file_path = "data_clean/test.crash"
reviews = read_crash_file(file_path)

# Tiền xử lý dữ liệu
reviews_processed = [normalize_text(review) for review in reviews]

# Dự đoán với mô hình
predictions = clf.predict(reviews_processed)

# Lưu kết quả vào DataFrame
results = pd.DataFrame({"review": reviews, "predicted_label": predictions})

# Chuyển kết quả thành chuỗi CSV
csv_result = results.to_csv(index=False)

# Ghi kết quả vào HDFS
output_path = "/user/ASUS/tri/sentiment_dicts/predicted_sentiment.csv"  # Đường dẫn tới thư mục HDFS
with client.write(output_path, overwrite=True) as writer:
    writer.write(
        csv_result.encode("utf-8")
    )  # Ghi dữ liệu vào HDFS dưới dạng byte (đã mã hóa UTF-8)

print(f"Predicted results saved to {output_path} in HDFS")
