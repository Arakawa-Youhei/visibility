import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import clip


# =========================
#  VGG19 feature extractor
# =========================

class VGG19FeatureExtractor(nn.Module):
    """VGG19 の中間特徴 (デフォルト: relu3_4) を返すモジュール。"""
    def __init__(self, layer_name: str = "relu3_4"):
        super().__init__()
        # PyTorch >= 1.13 を想定
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        # VGG layer index -> 名前 の対応表
        name_map = {
            1: 'relu1_1',
            3: 'relu1_2',
            6: 'relu2_1',
            8: 'relu2_2',
            11: 'relu3_1',
            13: 'relu3_2',
            15: 'relu3_3',
            17: 'relu3_4',
            20: 'relu4_1',
        }
        max_idx = None
        for idx, name in name_map.items():
            if name == layer_name:
                max_idx = idx
                break
        if max_idx is None:
            raise ValueError(f"Unknown layer_name: {layer_name}")

        self.features = vgg[:max_idx + 1].eval()
        for p in self.features.parameters():
            p.requires_grad = False

        # VGG 用の mean/std（ImageNet）
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W], 0-1, RGB
        """
        x = (x - self.mean) / self.std
        return self.features(x)


# =========================
#  Contextual distance
# =========================

def _flatten_feat(feat: torch.Tensor) -> torch.Tensor:
    """[B,C,H,W] -> [B,C,N]"""
    B, C, H, W = feat.shape
    return feat.view(B, C, H * W)


@torch.no_grad()
def contextual_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    vgg_extractor: VGG19FeatureExtractor,
    h: float = 0.5,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Contextual distance between x and y.

    Args:
        x, y: [B,3,H,W], 0-1, RGB
        vgg_extractor: VGG19FeatureExtractor インスタンス
        h: 温度パラメータ（小さいほどシャープ）
    Returns:
        dist: [B] 各バッチ要素の距離（小さいほど似ている）
    """
    device = next(vgg_extractor.parameters()).device
    x = x.to(device)
    y = y.to(device)

    fx = vgg_extractor(x)  # [B,C,Hf,Wf]
    fy = vgg_extractor(y)

    fx = _flatten_feat(fx)  # [B,C,Nx]
    fy = _flatten_feat(fy)  # [B,C,Ny]

    # L2 正規化
    fx = F.normalize(fx, p=2, dim=1)
    fy = F.normalize(fy, p=2, dim=1)

    B, C, Nx = fx.shape
    _, _, Ny = fy.shape

    # [B,Nx,C], [B,Ny,C]
    fx_t = fx.permute(0, 2, 1)
    fy_t = fy.permute(0, 2, 1)

    # 類似度行列 sim[b, j, i] = dot(fy_j, fx_i)
    sim = torch.bmm(fy_t, fx_t.transpose(1, 2))  # [B,Ny,Nx]

    # 数値安定化：i 方向に最大を引く
    sim_max, _ = sim.max(dim=2, keepdim=True)  # [B,Ny,1]
    sim_norm = sim - sim_max

    # 温度 h を使って softmax 的な重み
    w = torch.exp(sim_norm / (h + eps))  # [B,Ny,Nx]

    w_sum = w.sum(dim=2, keepdim=True) + eps
    cx_ij = w / w_sum  # [B,Ny,Nx]

    # 各 j について最も似ている i を取る
    cx_j, _ = cx_ij.max(dim=2)  # [B,Ny]

    # 全画素平均 → contextual similarity
    cx = cx_j.mean(dim=1)  # [B]

    # 距離：1 - CX（小さいほど似ている）
    dist = 1.0 - cx
    return dist


# =========================
#  CLIP-Score
# =========================

class CLIPScorer(nn.Module):
    """
    2枚の画像 x, y (0-1, [B,3,H,W]) に対して
    CLIP のコサイン類似度を返すモジュール。
    """
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        super().__init__()
        self.model, _ = clip.load(model_name, device=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # CLIP の画像前処理（tensor 前提）
        self.preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x, y: [B,3,H,W], 0-1, RGB
        Returns:
            sim: [B] コサイン類似度（大きいほど近い）
        """
        device = next(self.model.parameters()).device
        x = x.to(device)
        y = y.to(device)

        x_p = self.preprocess(x)
        y_p = self.preprocess(y)

        fx = self.model.encode_image(x_p)
        fy = self.model.encode_image(y_p)

        fx = F.normalize(fx, p=2, dim=-1)
        fy = F.normalize(fy, p=2, dim=-1)

        sim = (fx * fy).sum(dim=-1)  # [B]
        return sim


# =========================
#  画像ユーティリティ
# =========================

def load_image_as_tensor(path: str, img_size: int = 256) -> torch.Tensor:
    """
    画像ファイルを読み込んで [1,3,H,W], 0-1, RGB の tensor に変換。
    """
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # 0-1, [3,H,W]
    ])
    t = tfm(img).unsqueeze(0)  # [1,3,H,W]
    return t


def find_first_image(dir_path: str) -> str | None:
    """ディレクトリ内の最初の画像ファイルを返す（なければ None）。"""
    for f in sorted(os.listdir(dir_path)):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            return os.path.join(dir_path, f)
    return None


def list_images(dir_path: str):
    """ディレクトリ内のすべての画像ファイルパスを返す。"""
    return [
        os.path.join(dir_path, f)
        for f in sorted(os.listdir(dir_path))
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]


# =========================
#  メイン処理（1オブジェクト前提）
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir", type=str, required=True,
                        help="参照ビュー画像を1枚だけ入れたフォルダ")
    parser.add_argument("--gen_dir", type=str, required=True,
                        help="新規ビュー画像を複数枚入れたフォルダ")
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' か 'cpu'")
    parser.add_argument("--img_size", type=int, default=256,
                        help="評価前にリサイズする画像サイズ")
    parser.add_argument("--out_txt", type=str, default="metrics_single_object.txt",
                        help="結果を書き出すテキストファイル名")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 参照ビュー 1枚
    ref_path = find_first_image(args.ref_dir)
    if ref_path is None:
        print("ref_dir に画像が見つかりませんでした。")
        return

    # 新規ビュー 複数
    novel_paths = list_images(args.gen_dir)
    if not novel_paths:
        print("gen_dir に新規ビュー画像が見つかりませんでした。")
        return

    print(f"Reference image : {os.path.basename(ref_path)}")
    print(f"Novel view count: {len(novel_paths)}")

    # モデル
    vgg_extractor = VGG19FeatureExtractor(layer_name="relu3_4").to(device)
    clip_scorer = CLIPScorer(model_name="ViT-B/32", device=str(device))

    # 参照画像を一度だけ読み込み
    ref_img = load_image_as_tensor(ref_path, img_size=args.img_size)

    cx_values = []
    clip_values = []

    out_path = args.out_txt
    f = open(out_path, "w", encoding="utf-8")
    # type は sample / global_mean
    f.write("type,filename,contextual_distance,clip_score\n")

    for idx, novel_path in enumerate(novel_paths):
        novel_img = load_image_as_tensor(novel_path, img_size=args.img_size)

        # スコア計算
        cx = contextual_distance(ref_img, novel_img, vgg_extractor=vgg_extractor)
        cs = clip_scorer(ref_img, novel_img)
        cx_val = cx.item()
        cs_val = cs.item()

        cx_values.append(cx_val)
        clip_values.append(cs_val)

        filename = os.path.basename(novel_path)
        print(f"[{idx}] {filename}")
        print(f"  Contextual distance: {cx_val:.4f}")
        print(f"  CLIP-Score         : {cs_val:.4f}")

        # 各ビューごとの行
        f.write(f"sample,{filename},{cx_val:.6f},{cs_val:.6f}\n")

    # 全ビュー平均
    cx_mean = sum(cx_values) / len(cx_values)
    clip_mean = sum(clip_values) / len(clip_values)

    print("\n==== Mean over all novel views ====")
    print(f"Contextual distance (↓): {cx_mean:.4f}")
    print(f"CLIP-Score (↑)        : {clip_mean:.4f}")

    f.write(f"global_mean,ALL,{cx_mean:.6f},{clip_mean:.6f}\n")
    f.close()

    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
