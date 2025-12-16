
from torch import nn
from common import OCBAM # Đảm bảo bạn có file common.py chứa OCBAM

class Dual(nn.Module):
    """
    Kiến trúc CNN-2D kết hợp với CBAM để phân loại cảm xúc từ MFCC.
    Luồng xử lý: CNN trích xuất đặc trưng -> CBAM tinh chỉnh đặc trưng -> Classifier phân loại.
    """
    def __init__(self, num_classes=5): # Thêm tham số num_classes để linh hoạt
        super(Dual, self).__init__()

        # --- Khối trích xuất đặc trưng CNN ---
        self.feature_extractor = nn.Sequential(
            # Lớp 1
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            # Lớp 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            # Lớp 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4)),
            
            # Lớp 4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Sử dụng Adaptive Pooling để xử lý các kích thước đầu vào khác nhau
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # --- Cơ chế Attention ---
        # CBAM sẽ hoạt động trên đầu ra của khối CNN cuối cùng (256 kênh)
        self.cbam = OCBAM(256)

        # --- Khối phân loại (Classifier) ---
        self.classifier = nn.Sequential(
            nn.Flatten(), # Làm phẳng đầu ra từ (batch, 256, 1, 1) thành (batch, 256)
            nn.Dropout(p=0.5), # Thêm Dropout để chống overfitting
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes) # Đầu ra là số lớp
        )

    def forward(self, mfcc):
        # 1. Trích xuất đặc trưng bằng CNN
        features = self.feature_extractor(mfcc)
        
        # 2. Áp dụng Attention để tinh chỉnh đặc trưng
        refined_features = self.cbam(features)
        
        # 3. Phân loại
        output = self.classifier(refined_features)
        
        return output