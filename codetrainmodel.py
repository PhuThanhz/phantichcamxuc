# -*- coding: utf-8 -*-
from __future__ import print_function
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
import pandas as pd
from pyvi import ViTokenizer
import re
import os
from hdfs import InsecureClient
import codecs
import joblib
import io

# Kết nối đến HDFS
client = InsecureClient("http://localhost:50070")  # Thay đổi URL HDFS nếu cần

# Đường dẫn tới các file từ điển trên HDFS
DICT_PATH = "/user/ASUS/tri/sentiment_dicts/"
path_nag = os.path.join(DICT_PATH, "nag.txt")
path_pos = os.path.join(DICT_PATH, "pos.txt")
path_not = os.path.join(DICT_PATH, "not.txt")
path_sentiwordnet = os.path.join(DICT_PATH, "VietSentiWordNet_ver1.0.txt")


# Hàm đọc tệp từ HDFS mà không cần giải mã như chuỗi văn bản
def read_from_hdfs(path):
    with client.read(path) as reader:
        return reader.read()  # Trả về dữ liệu byte mà không giải mã


# Đọc từ điển cảm xúc từ HDFS
# Giải mã dữ liệu byte thành chuỗi trước khi xử lý
nag_list = read_from_hdfs(path_nag).decode("utf-8").splitlines()
pos_list = read_from_hdfs(path_pos).decode("utf-8").splitlines()
not_list = read_from_hdfs(path_not).decode("utf-8").splitlines()

# Đọc từ điển VietSentiWordNet từ HDFS
sentiwordnet_dict = {}
for line in read_from_hdfs(path_sentiwordnet).decode("utf-8").splitlines():
    if line.startswith("#") or not line.strip():
        continue
    parts = line.split("\t")
    word = parts[1]
    pos_score = float(parts[2])
    neg_score = float(parts[3])
    sentiwordnet_dict[word] = (pos_score, neg_score)

# Hàm loại bỏ dấu
VN_CHARS = (
    "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđð"
    + "ẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸÐĐ"
)
NO_VN_CHARS = (
    "a" * 17
    + "o" * 17
    + "e" * 11
    + "u" * 11
    + "i" * 5
    + "y" * 5
    + "d" * 2
    + "A" * 17
    + "O" * 17
    + "E" * 11
    + "U" * 11
    + "I" * 5
    + "Y" * 5
    + "D" * 2
)


def no_marks(s):
    __replaces_dict = dict(zip(VN_CHARS, NO_VN_CHARS))
    return re.sub(
        "|".join(__replaces_dict.keys()), lambda m: __replaces_dict[m.group(0)], s
    )


# Hàm chuẩn hóa văn bản, tích hợp VietSentiWordNet
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = ViTokenizer.tokenize(text)
    texts = text.split()
    processed_text = []

    for i, word in enumerate(texts):
        if word in not_list:
            if i < len(texts) - 1:
                if texts[i + 1] in pos_list:
                    processed_text.append("notpos")
                    continue
                elif texts[i + 1] in nag_list:
                    processed_text.append("notnag")
                    continue
        if word in pos_list:
            processed_text.append("positive")
        elif word in nag_list:
            processed_text.append("negative")
        elif word in sentiwordnet_dict:  # Sử dụng điểm cảm xúc từ VietSentiWordNet
            pos_score, neg_score = sentiwordnet_dict[word]
            if pos_score > neg_score:
                processed_text.append("positive")
            elif neg_score > pos_score:
                processed_text.append("negative")
        else:
            processed_text.append(word)

    return " ".join(processed_text)


# Tải dữ liệu và xử lý các giá trị NaN
def load_data_from_hdfs(filename):
    # Đọc dữ liệu từ HDFS
    data = read_from_hdfs(filename)

    # Lưu tạm dữ liệu vào tệp Excel cục bộ
    with open("temp_file.xlsx", "wb") as f:
        f.write(data)

    # Đọc tệp Excel vào pandas DataFrame
    df = pd.read_excel("temp_file.xlsx")

    # Đảm bảo cột dữ liệu đúng
    df.columns = ["id", "label", "review"]

    # Tiền xử lý cột review
    df["review"] = df["review"].fillna("").astype(str).apply(normalize_text)

    return df


# Chuẩn bị dữ liệu từ bigfile-train.xlsx trên HDFS
TRAIN_DATA_PATH = os.path.join(DICT_PATH, "bigfile-train.xlsx")
train_data = load_data_from_hdfs(TRAIN_DATA_PATH)
stop_ws = ["rằng", "thì", "là", "mà"]

# Huấn luyện mô hình
X_train, X_test, y_train, y_test = train_test_split(
    train_data.review, train_data.label, test_size=0.3, random_state=42
)

clf = Pipeline(
    [
        (
            "vect",
            CountVectorizer(
                ngram_range=(1, 2),
                stop_words=stop_ws,
                max_df=0.85,
                min_df=1,
            ),
        ),
        ("tfidf", TfidfTransformer(sublinear_tf=True)),
        ("clf", LinearSVC(C=1)),
    ]
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Classification Report:\n", metrics.classification_report(y_test, y_pred))

# Cross-validation
cross_score = cross_val_score(clf, X_train, y_train, cv=5)
print("Cross-validation score:", cross_score.mean())

# Lưu mô hình
MODEL_PATH = "sentiment_analysis_modelvipz.pkl"
joblib.dump(clf, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
