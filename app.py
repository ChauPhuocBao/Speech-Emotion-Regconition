import streamlit as st
import torch
import librosa
import numpy as np
import random  # Thư viện để chọn nhạc ngẫu nhiên
from model import Dual
from common import OCBAM

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================

# Cấu hình thiết bị (Demo nên dùng CPU để tránh lỗi CUDA out of memory)
DEVICE = torch.device("cpu")
MAX_LEN = 100  # Độ dài chuỗi MFCC (Phải khớp với lúc train)

# Danh sách nhãn cảm xúc
# LƯU Ý: Thứ tự này phải khớp với thứ tự folder lúc bạn train (thường là alphabet)
EMO_CLASSES = {
    0: "Angry (Tức giận)",
    1: "Fear (Sợ hãi)",
    2: "Happiness (Vui vẻ)",
    3: "Neutral (Bình thường)",
    4: "Sadness (Buồn bã)"
}

# Danh sách nhạc gợi ý theo cảm xúc (Link YouTube)
MUSIC_PLAYLIST = {
    0: [ # Angry -> Nhạc thiền, nhẹ nhàng để hạ hỏa
        "https://www.youtube.com/watch?v=lFcSrYw-ARY", # Mozart
        "https://www.youtube.com/watch?v=1ZYbU82GVz4", # Piano thư giãn
        "https://www.youtube.com/watch?v=hTPX5X-6M5o"  # Tiếng mưa rơi
    ],
    1: [ # Fear -> Nhạc trấn an, tích cực, không lời
        "https://www.youtube.com/watch?v=9p_DszJ1q90", # Nhạc tần số chữa lành
        "https://www.youtube.com/watch?v=neV3EPgvZ3g"  # Disney Piano
    ],
    2: [ # Happiness -> Nhạc sôi động, Pop, EDM
        "https://www.youtube.com/watch?v=09R8_2nJtjg", # Maroon 5 - Sugar
        "https://www.youtube.com/watch?v=OPf0YbXqDm0", # Uptown Funk
        "https://www.youtube.com/watch?v=ru0K8uYEZWw"  # Justin Timberlake
    ],
    3: [ # Neutral -> Nhạc Lofi, Chill để làm việc/học tập
        "https://www.youtube.com/watch?v=jfKfPfyJRdk", # Lofi Girl
        "https://www.youtube.com/watch?v=5qx7yNs4UoA", # Chill R&B
    ],
    4: [ # Sadness -> Nhạc buồn đồng cảm hoặc chữa lành
        "https://www.youtube.com/watch?v=4N3N1MlvVc4", # Nhạc không lời buồn
        "https://www.youtube.com/watch?v=RgKAFK5djSk", # See You Again
        "https://www.youtube.com/watch?v=RBumgq5yVrA"  # Let Her Go
    ]
}

# ==========================================
# 2. CÁC HÀM XỬ LÝ (BACKEND)
# ==========================================

@st.cache_resource
def load_model():
    """
    Load model một lần duy nhất và lưu vào Cache để chạy nhanh hơn.
    """
    try:
        # Khởi tạo kiến trúc model
        # num_classes=5 dựa trên danh sách EMO_CLASSES ở trên
        model = Dual(num_classes=len(EMO_CLASSES))
        
        # Load trọng số đã train
        model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
        
        # Chuyển sang chế độ đánh giá (không train nữa)
        model.to(DEVICE)
        model.eval()
        return model
    except FileNotFoundError:
        st.error("❌ Lỗi: Không tìm thấy file 'best_model.pth'. Vui lòng kiểm tra lại thư mục.")
        return None
    except Exception as e:
        st.error(f"❌ Lỗi khi khởi tạo model: {e}")
        return None

def preprocess_audio(file_path):
    """
    Biến file âm thanh thành Tensor đầu vào cho mô hình
    """
    try:
        # Load file (chỉ lấy 3 giây đầu để xử lý nhanh và tránh lỗi bộ nhớ)
        y, sr = librosa.load(file_path, sr=16000, duration=3.0)
        
        # Nếu file quá ngắn (dưới 0.5s), bỏ qua hoặc padding nhiều sẽ không tốt
        if len(y) < 1000:
            return None

        # Trích xuất MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # Padding hoặc cắt ngắn cho đúng chuẩn MAX_LEN = 100
        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_LEN]
            
        # Chuyển thành Tensor 4 chiều: (Batch, Channel, Height, Width)
        # Kích thước: (1, 1, 40, 100)
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return mfcc_tensor
        
    except Exception as e:
        st.error(f"Lỗi xử lý âm thanh: {e}")
        return None

# ==========================================
# 3. GIAO DIỆN CHÍNH (FRONTEND)
# ==========================================

def main():
    # Cấu hình trang web
    st.set_page_config(page_title="Emotion Music AI", page_icon="🎵", layout="wide")
    
    st.title("🎙️ Hệ thống Gợi ý Nhạc theo Cảm xúc")
    st.write("Sử dụng mô hình AI (Dual-CNN) để phân tích giọng nói và đề xuất bài hát phù hợp.")
    st.divider()

    # Load model (chỉ chạy 1 lần đầu tiên)
    with st.spinner("Đang khởi động hệ thống AI..."):
        model = load_model()
        
    # Nếu model lỗi thì dừng chương trình
    if model is None:
        st.stop()

    # Chia giao diện thành 2 cột: Cột Trái (Input) - Cột Phải (Kết quả)
    col_input, col_result = st.columns([1, 1.5], gap="large")

    # --- CỘT TRÁI: NHẬP LIỆU ---
    with col_input:
        st.header("1. Nhập giọng nói")
        st.info("Hãy nói một câu thể hiện cảm xúc của bạn (khoảng 3 giây).")
        
        # Tab chọn: Tải file hoặc Ghi âm
        tab1, tab2 = st.tabs(["📂 Tải file lên", "🎙️ Ghi âm trực tiếp"])
        
        file_to_process = None
        
        with tab1:
            uploaded_file = st.file_uploader("Chọn file .wav hoặc .mp3", type=["wav", "mp3"])
            if uploaded_file:
                file_to_process = uploaded_file
        
        with tab2:
            # Kiểm tra xem Streamlit có hỗ trợ ghi âm không (bản mới mới có)
            if hasattr(st, "audio_input"):
                audio_buffer = st.audio_input("Nhấn nút đỏ để ghi âm")
                if audio_buffer:
                    file_to_process = audio_buffer
            else:
                st.warning("Phiên bản Streamlit của bạn cũ, chưa hỗ trợ ghi âm trực tiếp. Vui lòng dùng tính năng tải file.")

        # Nút bấm xử lý
        if file_to_process:
            st.audio(file_to_process, format="audio/wav")
            process_btn = st.button("Phân tích Cảm xúc", type="primary", use_container_width=True)
        else:
            process_btn = False

    # --- CỘT PHẢI: KẾT QUẢ ---
    with col_result:
        st.header("2. Kết quả & Âm nhạc")
        
        if process_btn and file_to_process:
            with st.spinner("AI đang lắng nghe và phân tích..."):
                # 1. Tiền xử lý
                input_tensor = preprocess_audio(file_to_process)
                
                if input_tensor is not None:
                    # Đưa lên thiết bị (CPU)
                    input_tensor = input_tensor.to(DEVICE)
                    
                    # 2. Dự đoán
                    with torch.no_grad():
                        output = model(input_tensor)
                        # Tính xác suất (Softmax)
                        probs = torch.nn.functional.softmax(output, dim=1)
                        confidence, predicted = torch.max(probs, 1)
                    
                    # 3. Lấy thông tin kết quả
                    idx = predicted.item()      # ID lớp (0, 1, 2...)
                    label = EMO_CLASSES.get(idx, "Không xác định")
                    score = confidence.item() * 100
                    
                    # --- HIỂN THỊ KẾT QUẢ ---
                    st.success(f"### Cảm xúc nhận diện: {label}")
                    st.caption(f"Độ tin cậy của AI: {score:.2f}%")
                    
                    # Vẽ biểu đồ xác suất
                    chart_data = {
                        name: prob.item() 
                        for name, prob in zip(EMO_CLASSES.values(), probs[0])
                    }
                    st.bar_chart(chart_data, height=200)

                    # --- GỢI Ý NHẠC (FEATURE MỚI) ---
                    st.divider()
                    st.subheader(f"🎵 Gợi ý nhạc cho tâm trạng '{label}'")
                    
                    if idx in MUSIC_PLAYLIST:
                        # Chọn ngẫu nhiên 1 bài hát
                        song_url = random.choice(MUSIC_PLAYLIST[idx])
                        
                        st.write("Dựa trên tâm trạng của bạn, AI đề xuất bài hát này:")
                        # Phát video YouTube
                        st.video(song_url)
                    else:
                        st.info("Chưa có dữ liệu nhạc cho cảm xúc này.")
                        
                else:
                    st.error("File âm thanh không hợp lệ hoặc quá ngắn/quá dài.")
        
        elif not file_to_process:
            st.info("👈 Vui lòng chọn file hoặc ghi âm bên cột trái để bắt đầu.")

# Chạy ứng dụng
if __name__ == "__main__":
    main()