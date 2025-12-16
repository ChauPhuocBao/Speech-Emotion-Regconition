import streamlit as st
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from model import Dual
from common import OCBAM

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
# LƯU Ý: Bạn cần sửa lại danh sách này đúng theo thứ tự alphabet
# của các thư mục trong dataset VNEMOS lúc bạn train.
# Ví dụ: nếu folder là 'angry', 'fear', 'happiness', 'neutral', 'sadness'
# thì thứ tự sẽ là:
EMO_CLASSES = {
    0: "Angry (Tức giận)",
    1: "Fear (Sợ hãi)",
    2: "Happiness (Vui vẻ)",
    3: "Neutral (Bình thường)",
    4: "Sadness (Buồn bã)"
}

DEVICE = torch.device("cpu") # Chạy demo trên CPU cho ổn định
MAX_LEN = 100 # Độ dài chuỗi MFCC (phải khớp với lúc train)

# ==========================================
# 2. CÁC HÀM XỬ LÝ
# ==========================================
@st.cache_resource
def load_model():
    """Load model một lần duy nhất để dùng mãi mãi"""
    # Khởi tạo model với số lớp là 5 (như kết quả debug của bạn)
    model = Dual(num_classes=len(EMO_CLASSES))
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Lỗi load model: {e}")
        return None

def preprocess_audio(file_path):
    """Biến file âm thanh thành Tensor đầu vào cho mô hình"""
    try:
        # Load file (chỉ lấy 3 giây đầu để xử lý nhanh)
        y, sr = librosa.load(file_path, sr=16000, duration=3.0)
        
        # Trích xuất MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # Padding hoặc cắt ngắn cho đúng chuẩn MAX_LEN
        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_LEN]
            
        # Chuyển thành Tensor 4 chiều: (Batch, Channel, Height, Width)
        # (1, 1, 40, 100)
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return mfcc_tensor
    except Exception as e:
        st.error(f"Lỗi xử lý âm thanh: {e}")
        return None

# ==========================================
# 3. GIAO DIỆN CHÍNH (MAIN APP)
# ==========================================
def main():
    st.set_page_config(page_title="Emotion AI Demo", page_icon="🎙️")
    
    st.title("🎙️ Nhận Diện Cảm Xúc Giọng Nói (VNEMOS)")
    st.write("Hệ thống sử dụng mô hình Deep Learning (CNN + Attention) để phân tích giọng nói tiếng Việt.")
    
    # Load model
    with st.spinner("Đang khởi động AI..."):
        model = load_model()
        
    if model is None:
        st.warning("⚠️ Không thể chạy ứng dụng do lỗi Model.")
        st.stop()

    st.divider()

    # --- Cột trái: Input ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Đầu vào")
        uploaded_file = st.file_uploader("Tải lên file ghi âm (.wav)", type=["wav", "mp3"])
        
        # (Tùy chọn) Ghi âm trực tiếp - Chỉ hoạt động trên Streamlit mới nhất
        audio_input = st.audio_input("Hoặc ghi âm trực tiếp") if hasattr(st, "audio_input") else None

    # Xác định file để xử lý
    file_to_process = uploaded_file if uploaded_file else audio_input

    # --- Cột phải: Kết quả ---
    with col2:
        st.subheader("2. Phân tích")
        
        if file_to_process:
            # Nghe lại file
            st.audio(file_to_process, format="audio/wav")
            
            if st.button("🚀 Chạy Mô hình", type="primary"):
                with st.spinner("Đang phân tích tín hiệu..."):
                    # 1. Tiền xử lý
                    input_tensor = preprocess_audio(file_to_process)
                    
                    if input_tensor is not None:
                        input_tensor = input_tensor.to(DEVICE)
                        
                        # 2. Dự đoán
                        with torch.no_grad():
                            output = model(input_tensor)
                            # Tính xác suất (Softmax)
                            probs = torch.nn.functional.softmax(output, dim=1)
                            confidence, predicted = torch.max(probs, 1)
                        
                        # 3. Hiển thị kết quả
                        idx = predicted.item()
                        label = EMO_CLASSES.get(idx, "Không xác định")
                        score = confidence.item() * 100
                        
                        # Hộp kết quả nổi bật
                        st.success(f"### Kết quả: {label}")
                        st.info(f"Độ tin cậy: **{score:.2f}%**")
                        
                        # 4. Vẽ biểu đồ cột xác suất
                        st.write("Chi tiết xác suất các lớp:")
                        chart_data = {
                            name: prob.item() 
                            for i, (name, prob) in enumerate(zip(EMO_CLASSES.values(), probs[0]))
                        }
                        st.bar_chart(chart_data)
        else:
            st.info("👈 Vui lòng tải file hoặc ghi âm để bắt đầu.")

if __name__ == "__main__":
    main()