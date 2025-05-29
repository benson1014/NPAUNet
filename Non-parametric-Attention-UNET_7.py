import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torchvision.transforms.functional as TF
import torch.nn.functional as F


#--------------------
# 參數設定
#--------------------
EPOCH = 50
BATCH_SIZE = 8
LR = 1e-4
LAMBDA_EDGE = 0.3  # Edge branch loss 權重
LAMBDA_IOU = 0.3
SOFTIOU = True     # True 表示用 SoftIOU； False 則用 Lovasz。
IMG_SIZE = 256
X_PATH = "skin-lesion/augmented_trainx/*.bmp"
Y_PATH = "skin-lesion/augmented_trainy/*.bmp"


# --------------------
# 資料集與增強
# --------------------
def augment_images(image, mask):
    angle = np.random.randint(-40, 40)
    image = TF.rotate(image, angle)
    mask = TF.rotate(mask, angle)
    if np.random.rand() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    return image, mask


class LesionDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform_img=None, transform_mask=None, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        mask = Image.open(self.mask_paths[idx]).convert("L").resize((IMG_SIZE, IMG_SIZE))
        if self.augment:
            image, mask = augment_images(image, mask)
        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)
        return image, mask

# --------------------
# Loss 與 helper
# --------------------
def sobel_edge(mask: torch.Tensor) -> torch.Tensor:
    """mask: [B,1,H,W] (0/1) → 返回同 shape 邊界 0/1"""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(mask, sobel_x, padding=1)
    gy = F.conv2d(mask, sobel_y, padding=1)
    edge = (gx.abs() + gy.abs()).clamp(0, 1)
    return edge

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        prob = prob.view(prob.size(0), -1)
        target = target.view(target.size(0), -1)
        inter = (prob * target).sum(1)
        dice = (2 * inter + self.smooth) / (prob.sum(1) + target.sum(1) + self.smooth)
        return 1 - dice.mean()

#----------
# SoftIOU Loss
#----------
class SoftIOULoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(SoftIOULoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(1)
        union = probs.sum(1) + targets.sum(1) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()

#----------
# Lovasz-Softmax Loss
#----------
def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1. - intersection / union
    if gt_sorted.numel() == 0:
        return torch.tensor([])
    return jaccard[1:] - jaccard[:-1]

def flatten_binary_scores(scores, labels):
    scores = scores.view(-1)
    labels = labels.view(-1)
    return scores, labels

class LovaszHingeLoss(nn.Module):
    def __init__(self, per_image=True):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image

    def forward(self, logits, labels):
        if self.per_image:
            loss = torch.mean(torch.stack([self.lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0)))
                                           for log, lab in zip(logits, labels)]))
        else:
            loss = self.lovasz_hinge_flat(*flatten_binary_scores(logits, labels))
        return loss

    def lovasz_hinge_flat(self, logits, labels):
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * signs)
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = lovasz_grad(gt_sorted)
        if grad.numel() == 0:
            return torch.tensor(0., device=logits.device)
        loss = torch.dot(F.relu(errors_sorted)[:grad.numel()], grad)
        return loss

class MaskEdgeLoss(nn.Module):
    def __init__(self, lambda_edge=LAMBDA_EDGE, use_softiou=True, lambda_iou=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.softiou = SoftIOULoss()
        self.lovasz = LovaszHingeLoss()
        self.lambda_edge = lambda_edge
        self.use_softiou = use_softiou
        self.lambda_iou = lambda_iou

    def forward(self, mask_logits, edge_logits, mask_gt):
        loss_mask = self.dice(mask_logits, mask_gt)
        edge_gt = sobel_edge(mask_gt)
        loss_edge = self.bce(edge_logits, edge_gt)
        
        # 加入 IOU-based Loss
        if self.use_softiou:
            loss_iou = self.softiou(mask_logits, mask_gt)
        else:
            loss_iou = self.lovasz(mask_logits, mask_gt)
        
        return loss_mask + self.lambda_edge * loss_edge + self.lambda_iou * loss_iou


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


# 搭配 StatisticalAttention
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
# 評估指標
# --------------------
def iou_pytorch(prob, target, smooth=1e-6):
    pred = (prob > 0.5).float()
    inter = (pred * target).sum(dim=[1, 2, 3])
    union = pred.sum(dim=[1, 2, 3]) + target.sum(dim=[1, 2, 3]) - inter
    return ((inter + smooth) / (union + smooth)).mean().item()

def dice_coef_pytorch(prob, target, smooth=1e-6):
    pred = (prob > 0.5).float()
    inter = (pred * target).sum(dim=[1, 2, 3])
    return ((2 * inter + smooth) / (pred.sum(dim=[1, 2, 3]) + target.sum(dim=[1, 2, 3]) + smooth)).mean().item()

def precision_pytorch(prob, target, smooth=1e-6):
    pred = (prob > 0.7).float()
    tp = (pred * target).sum(dim=[1, 2, 3])
    pp = pred.sum(dim=[1, 2, 3])
    return ((tp + smooth) / (pp + smooth)).mean().item()

def recall_pytorch(prob, target, smooth=1e-6):
    pred = (prob > 0.5).float()
    tp = (pred * target).sum(dim=[1, 2, 3])
    pp = target.sum(dim=[1, 2, 3])
    return ((tp + smooth) / (pp + smooth)).mean().item()

def accuracy_pytorch(prob, target):
    pred = (prob > 0.5).float()
    return (pred == target).float().mean().item()

def plot_training_history(train_losses, val_losses, train_ious, val_ious, train_dices, val_dices,
                             train_precisions, val_precisions, train_recalls, val_recalls):
    epochs = range(1, len(train_losses) + 1)
    fig, axs = plt.subplots(2, 3, figsize=(60, 35))
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    # 設定共通字型與線條
    font_size = 10
    line_width = 1
    train_color = '#1f77b4'
    val_color = '#ff7f0e'

    metrics = [
        ('Loss', train_losses, val_losses, 'Loss'),
        ('IoU', train_ious, val_ious, 'IoU'),
        ('Dice_coef', train_dices, val_dices, 'Dice'),
        ('Precision', train_precisions, val_precisions, 'Precision'),
        ('Recall', train_recalls, val_recalls, 'Recall')
    ]

    positions = [(0,0), (0,1), (0,2), (1,0), (1,1)]
    for i, (title, train_values, val_values, ylabel) in enumerate(metrics):
        ax = axs[positions[i]]
        ax.plot(epochs, train_values, color=train_color, linewidth=line_width, label=f"Train last value: {train_values[-1]:.4f}")
        ax.plot(epochs, val_values, color=val_color, linewidth=line_width, label=f"Val last value: {val_values[-1]:.4f}")
        ax.set_title(title, fontsize=font_size)
        ax.set_xlabel('Epochs', fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.grid(True)
        # 圖例位置右下
        ax.legend(loc='lower right', fontsize=font_size, frameon=False)

    axs[1, 2].axis('off') 

    plt.suptitle('Training History UNET', fontsize=16)
    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('training_history.png')
    plt.show()


# --------------------
# 評估函式
# --------------------
def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    running_loss = iou_sum = dice_sum = precision_sum = recall_sum = accuracy_sum = count = 0
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            mask_logits, edge_logits = model(imgs)
            loss = criterion(mask_logits, edge_logits, masks)
            prob = torch.sigmoid(mask_logits)
            bsz = imgs.size(0)
            running_loss += loss.item() * bsz
            iou_sum += iou_pytorch(prob, masks) * bsz
            dice_sum += dice_coef_pytorch(prob, masks) * bsz
            precision_sum += precision_pytorch(prob, masks) * bsz
            recall_sum += recall_pytorch(prob, masks) * bsz
            accuracy_sum += accuracy_pytorch(prob, masks) * bsz
            count += bsz
    return {
        'loss': running_loss / count,
        'iou': iou_sum / count,
        'dice': dice_sum / count,
        'precision': precision_sum / count,
        'recall': recall_sum / count,
        'accuracy': accuracy_sum / count
    }

# --------------------
# train
# --------------------
def train(model, train_loader, val_loader, device, epochs=30, lr=5e-5):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = MaskEdgeLoss(lambda_edge=LAMBDA_EDGE, use_softiou=SOFTIOU, lambda_iou=LAMBDA_IOU) # 0.3~0.5

    best_val_dice = 0.0 
    best_model_path = "./checkpoint/unet_stat_attention_best.pth"

    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    train_dices, val_dices = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, masks in loop:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            mask_logits, edge_logits = model(imgs)
            loss = criterion(mask_logits, edge_logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=train_loss / len(train_loader))

        train_metrics = evaluate_model(model, train_loader, device, criterion)
        val_metrics = evaluate_model(model, val_loader, device, criterion)

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_metrics['loss'])
        train_ious.append(train_metrics['iou'])
        val_ious.append(val_metrics['iou'])
        train_dices.append(train_metrics['dice'])
        val_dices.append(val_metrics['dice'])
        train_precisions.append(train_metrics['precision'])
        val_precisions.append(val_metrics['precision'])
        train_recalls.append(train_metrics['recall'])
        val_recalls.append(val_metrics['recall'])

        val_dice = val_metrics['dice']
        print(f"Epoch {epoch+1} - Train Loss: {train_loss/len(train_loader):.4f} - Val Dice: {val_dice:.4f}")

        # 檢查是否是最佳模型，若是則儲存
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch+1} with Val Dice: {best_val_dice:.4f}")

    print(f"Training complete. Best validation Dice: {best_val_dice:.4f}")

    # 最後畫圖
    plot_training_history(train_losses, val_losses, train_ious, val_ious, train_dices, val_dices,
                          train_precisions, val_precisions, train_recalls, val_recalls)

# --------------------
# 可視化
# --------------------
def visualize_prediction(model, dataloader, device, num_samples=5):
    model.eval()
    count = 0
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            mask_logits, _ = model(imgs)
            outputs = (torch.sigmoid(mask_logits) > 0.5).float()
            imgs_np = imgs.cpu().permute(0, 2, 3, 1).numpy()
            masks_np = masks.cpu().numpy()
            outputs_np = outputs.cpu().numpy()
            for i in range(imgs_np.shape[0]):
                if count >= num_samples:
                    return
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(imgs_np[i]); axs[0].set_title("Original Image")
                axs[1].imshow(masks_np[i][0], cmap='gray'); axs[1].set_title("Ground Truth")
                axs[2].imshow(outputs_np[i][0], cmap='gray'); axs[2].set_title("Prediction")
                for ax in axs:
                    ax.axis('off')
                plt.tight_layout(); plt.show()
                count += 1

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    numbers = re.compile(r'(\d+)')
    def numerical_sort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    x_paths = sorted(glob.glob(X_PATH), key=numerical_sort)
    y_paths = sorted(glob.glob(Y_PATH), key=numerical_sort)
    x_trainval, x_test, y_trainval, y_test = train_test_split(x_paths, y_paths, test_size=0.25, random_state=101)
    x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.2, random_state=101)
    
    transform_img = transforms.ToTensor()
    transform_mask = transforms.ToTensor()
    
    train_dataset = LesionDataset(x_train, y_train, transform_img, transform_mask, augment=True)
    val_dataset = LesionDataset(x_val, y_val, transform_img, transform_mask)
    test_dataset = LesionDataset(x_test, y_test, transform_img, transform_mask)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet_EdgeBranch_AttentionGate().to(device)

    train(model, train_loader, val_loader, device, epochs=EPOCH, lr=LR)

    #torch.save(model.state_dict(), "./pth/unet_stat_attention.pth")
    #criterion = nn.BCELoss() # BCELoss BCEWithLogitsLoss
    best_model_path = "./checkpoint/unet_stat_attention_best.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    criterion = MaskEdgeLoss(lambda_edge=LAMBDA_EDGE, use_softiou=SOFTIOU, lambda_iou=LAMBDA_IOU)


    for name, loader in zip(["Train", "Validation", "Test"], [train_loader, val_loader, test_loader]):
        metrics = evaluate_model(model, loader, device, criterion)
        print(f"\n{name} Set Metrics")
        print(f"Loss:      {metrics['loss']:.4f}")
        print(f"IOU:       {metrics['iou']*100:.2f}%")
        print(f"Dice:      {metrics['dice']*100:.2f}%")
        print(f"Precision: {metrics['precision']*100:.2f}%")
        print(f"Recall:    {metrics['recall']*100:.2f}%")
        print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")

    visualize_prediction(model, test_loader, device)
