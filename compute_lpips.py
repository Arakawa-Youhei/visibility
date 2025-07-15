import torch
import lpips
from PIL import Image
import torchvision.transforms as transforms
import argparse

# 引数の定義
parser = argparse.ArgumentParser()
parser.add_argument('--img1', type=str, required=True, help='画像1のパス（GTなど）')
parser.add_argument('--img2', type=str, required=True, help='画像2のパス（再構成画像など）')
args = parser.parse_args()

# LPIPSモデルのロード（デフォルトはAlexNet）
loss_fn = lpips.LPIPS(net='alex')  # 例: 'alex', 'vgg'

# 画像の読み込みと前処理
transform = transforms.Compose([
    transforms.Resize((256, 256)),     # 入力画像のサイズを統一
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # [-1, 1] にスケーリング
])

def load_image(path):
    image = Image.open(path).convert('RGB')
    return transform(image).unsqueeze(0)  # [1, 3, H, W]

img0 = load_image(args.img1)
img1 = load_image(args.img2)

# LPIPS値の計算
with torch.no_grad():
    dist = loss_fn(img0, img1)

print(f"LPIPS Distance: {dist.item():.4f}")
