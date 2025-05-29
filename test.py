import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import os
from itertools import chain
from PIL import ImageOps

# --------------------
# 載入模型架構
# --------------------
class StatisticalAttention(nn.Module):
    def __init__(self):
        super(StatisticalAttention, self).__init__()

    def forward(self, x):
        avg = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True)
        skew = ((x - avg) ** 3).mean(dim=[2, 3], keepdim=True) / (std + 1e-6) ** 3
        kurt = ((x - avg) ** 4).mean(dim=[2, 3], keepdim=True) / (std + 1e-6) ** 4
        stats = torch.cat([avg, std, skew, kurt], dim=1)
        weights = torch.softmax(stats.mean(dim=[2, 3]), dim=1).unsqueeze(-1).unsqueeze(-1)
        return x * weights.sum(dim=1, keepdim=True)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # 注意力加權後的 skip 特徵

class UNet_EdgeBranch_AttentionGate(nn.Module):
    def __init__(self):
        super(UNet_EdgeBranch_AttentionGate, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.pool = nn.MaxPool2d(2)
        self.attn = StatisticalAttention()
        self.attn1 = StatisticalAttention()
        self.attn2 = StatisticalAttention()
        self.attn3 = StatisticalAttention()
        self.attn4 = StatisticalAttention()

        self.conv1 = conv_block(3, 32)
        self.conv2 = conv_block(32, 64)
        self.conv3 = conv_block(64, 128)
        self.conv4 = conv_block(128, 256)
        self.conv5 = conv_block(256, 512)

        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up6_conv = conv_block(256, 256)   # 對應 u6
        self.conv6 = conv_block(512, 256)

        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up7_conv = conv_block(128, 128)   # 對應 u7
        self.conv7 = conv_block(256, 128)
        
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = conv_block(128, 64)

        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = conv_block(64, 32)

        self.final_mask = nn.Conv2d(32, 1, kernel_size=1)
        self.edge_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )

        # Attention Gates
        self.ag6 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.ag7 = AttentionGate(F_g=128, F_l=128, F_int=64)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool(c1)
        p1 = self.attn1(p1)

        c2 = self.conv2(p1)
        p2 = self.pool(c2)
        p2 = self.attn2(p2)

        c3 = self.conv3(p2)
        p3 = self.pool(c3)
        p3 = self.attn3(p3)

        c4 = self.conv4(p3)
        p4 = self.pool(c4)
        p4 = self.attn4(p4)

        c5 = self.conv5(p4)
        c5 = self.attn(c5)

        u6 = self.up6(c5)
        u6 = self.up6_conv(u6)   # conv block 處理上採樣輸出，增強語意匹配
        c4_attn = self.ag6(u6, c4)
        u6 = torch.cat([u6, c4_attn], dim=1)
        c6 = self.conv6(u6)

        u7 = self.up7(c6)
        u7 = self.up7_conv(u7)   # conv block 處理上採樣輸出，增強語意匹配
        c3_attn = self.ag7(u7, c3)
        u7 = torch.cat([u7, c3_attn], dim=1)
        c7 = self.conv7(u7)

        u8 = self.up8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.conv8(u8)

        u9 = self.up9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.conv9(u9)

        mask_logits = self.final_mask(c9)
        edge_logits = self.edge_head(c9)
        return mask_logits, edge_logits

# --------------------
# 定義前處理
# --------------------
IMG_SIZE = 256 
transform_img = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# --------------------
# 載入模型
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet_EdgeBranch_AttentionGate().to(device)

pth_path = './checkpoint/unet_stat_attention_best.pth' 
model.load_state_dict(torch.load(pth_path, map_location=device))
model.eval()

# --------------------
# 載入圖片進行預測
# --------------------
test_img_dir = 'image_test/'
output_dir = 'image/'
os.makedirs(output_dir, exist_ok=True)

numbers = re.compile(r'(\d+)')
def numerical_sort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

extensions = ['*.bmp', '*.png', '*.jpg', '*.jpeg']
img_paths = sorted(chain.from_iterable(glob.glob(os.path.join(test_img_dir, ext)) for ext in extensions), key=numerical_sort)

for img_path in img_paths:
    img_name = os.path.basename(img_path)
    img = Image.open(img_path).convert('RGB')
    
    img_resized = img.resize((512, 512), Image.BILINEAR)
    
    img_tensor = transform_img(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        mask_logits, _ = model(img_tensor)
        pred_mask = (torch.sigmoid(mask_logits) > 0.5).float().cpu().numpy()[0, 0]
    
    pred_mask_resized = Image.fromarray((pred_mask * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST)
    
    # 繪製
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_resized)
    plt.title('Input Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask_resized, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1)
    
    output_path = os.path.join(output_dir, f'pred_{os.path.splitext(img_name)[0]}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved prediction to {output_path}")

print("所有圖片預測完成！")
