import joblib
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from pyvi import ViTokenizer
import os

# Đường dẫn tới các file model
model_paths = {
    "Model 1": "sentiment_analysis_model.pkl",
    "Model 2": "sentiment_analysis_modelvip.pkl",
    "Model 3": "sentiment_model.pkl",
}

# Đường dẫn tới file dữ liệu test
DICT_PATH = "D:/EMOTION/sentiment_analysis_nal-master/sentiment_analysis_nal-master/sentiment_dicts/"
TEST_DATA_PATH = os.path.join(DICT_PATH, "bigfile-train.xlsx")

# Đọc tập dữ liệu test
test_data = pd.read_excel(TEST_DATA_PATH)
test_data.columns = ["id", "label", "review"]


# Hàm tiền xử lý văn bản
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = ViTokenizer.tokenize(text)
    return text


# Tiền xử lý dữ liệu review
test_data["review"] = test_data["review"].fillna("").astype(str).apply(normalize_text)
X_test = test_data["review"]
y_true = test_data["label"]

# So sánh các mô hình
for model_name, model_path in model_paths.items():
    # Tải model
    model = joblib.load(model_path)

    # Dự đoán và đánh giá
    y_pred = model.predict(X_test)
    print(f"Đánh giá cho {model_name}:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print(f"Accuracy của {model_name}: {accuracy_score(y_true, y_pred):.4f}")
    print("\n" + "=" * 50 + "\n")
