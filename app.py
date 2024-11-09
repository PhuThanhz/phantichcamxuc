import streamlit as st
import joblib
from pyvi import ViTokenizer
import re
import string


# Hàm chuẩn hóa văn bản, tương tự trong sourcecode.py
def normalize_text(text):
    text = re.sub(
        r"([A-Z])\1+", lambda m: m.group(1).upper(), text, flags=re.IGNORECASE
    )
    text = text.lower()

    replace_list = {"òa": "oà", "óa": "oá", "ảm": "oả"}  # Thêm các thay thế nếu cần

    for k, v in replace_list.items():
        text = text.replace(k, v)

    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    text = text.translate(translator)

    text = ViTokenizer.tokenize(text)
    return text


# Hàm tải mô hình đã huấn luyện
def load_model():
    try:
        # Tải mô hình đã lưu từ file sentiment_model.pkl
        model = joblib.load("sentiment_model.pkl")
        return model
    except FileNotFoundError:
        st.error(
            "Không tìm thấy mô hình sentiment_model.pkl. Hãy đảm bảo mô hình đã được huấn luyện và lưu."
        )
        return None


# Tạo giao diện với Streamlit
st.title("Ứng dụng Phân tích Cảm xúc")

st.write("Nhập đoạn văn bản để phân tích cảm xúc:")

# Nhập văn bản từ người dùng
user_input = st.text_area("Văn bản đầu vào:", "")

# Khi nhấn nút "Phân tích cảm xúc", tiến hành dự đoán
if st.button("Phân tích cảm xúc"):
    if user_input:
        model = load_model()  # Tải mô hình đã huấn luyện

        if model is not None:
            user_input_normalized = normalize_text(
                user_input
            )  # Chuẩn hóa văn bản đầu vào
            prediction = model.predict([user_input_normalized])  # Dự đoán cảm xúc

            # Hiển thị giá trị dự đoán
            st.write(f"Kết quả dự đoán từ mô hình: {prediction}")

            # Chuyển giá trị từ chuỗi sang số nguyên để so sánh (phòng trường hợp là chuỗi)
            prediction = int(prediction[0])

            # Kiểm tra kết quả dự đoán và hiển thị
            if prediction == 1:  # Nếu kết quả dự đoán là 1 (tiêu cực)
                st.write("Kết quả: Cảm xúc tiêu cực")
            elif prediction == 0:  # Nếu kết quả dự đoán là 0 (tích cực)
                st.write("Kết quả: Cảm xúc tích cực")
            else:
                st.write("Không xác định được cảm xúc.")
        else:
            st.write("Vui lòng tải lại mô hình.")
    else:
        st.write("Vui lòng nhập vào một đoạn văn bản.")
