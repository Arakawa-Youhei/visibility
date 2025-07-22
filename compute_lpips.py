import os
import glob
import lpips
import torch
import torchvision.transforms as transforms
from PIL import Image

# === 入力設定 ===
gt_dir = "/home/arakawa/HyperDreamer/exp/frog3c_s2/results/gt"
pred_dir = "/home/arakawa/HyperDreamer/exp/frog3c_s2/results/output"
pattern = "*.png"
output_file = "result_lpips.txt"
# ===============

def load_image(path, size=(256, 256)):
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(img).unsqueeze(0)

def compute_lpips(gt_dir, pred_dir, pattern='*.png', output_file='result_lpips.txt'):
    loss_fn = lpips.LPIPS(net='alex').cuda()
    gt_files = sorted(glob.glob(os.path.join(gt_dir, pattern)))
    pred_files = sorted(glob.glob(os.path.join(pred_dir, pattern)))

    scores = []

    with open(output_file, 'w') as f:
        for gt, pred in zip(gt_files, pred_files):
            img1 = load_image(gt).cuda()
            img2 = load_image(pred).cuda()
            d = loss_fn(img1, img2).item()
            line = f"[LPIPS] {os.path.basename(gt)} vs {os.path.basename(pred)}: {d:.4f}"
            print(line)
            f.write(line + '\n')
            scores.append(d)

        mean_score = sum(scores) / len(scores)
        summary = f"\n[LPIPS] Mean LPIPS: {mean_score:.4f}"
        print(summary)
        f.write(summary + '\n')

if __name__ == "__main__":
    compute_lpips(gt_dir, pred_dir, pattern, output_file)
