# -*- coding: utf-8 -*-
"""用 best 权重（ep5953, loss 1723.99）重新生成 A-E 图 + 客观质量指标。

A1/A2 重建（PSNR 在全量 5000 张上算）、B 潜空间采样、C 像素采样对照、
D 插值、E 重组。并打印量化指标辅助判断（当前对话无法直接读图）。
"""
import pickle
import sys
import numpy as np
import torch

sys.path.insert(0, r"d:\大三上的资料\假期学习记录\机器学习基础")
import vae_cat_full as V  # 复用 ConvVAE / 数据加载 / 网格拼接

OUT = r"d:\大三上的资料\假期学习记录\机器学习基础"
BEST = rf"{OUT}\vae_cat_v2_best.pt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_data():
    cats = V.load_cats(V.DATA)
    X = torch.from_numpy(V.to_img(cats))
    return X


def psnr(xr, x, maxv=1.0):
    mse = ((xr - x) ** 2).mean().item()
    return 10 * np.log10(maxv ** 2 / mse) if mse > 0 else 99.0


def nat_score(imgs):
    """自然度粗指标：通道均值越接近 0.5、梯度越大越接近真实图边缘。
    返回 (channel_mean_abs_dev, mean_grad) —— 越小越"偏灰/平淡"，越大越锐利。"""
    g = (imgs[:, :, 1:, :] - imgs[:, :, :-1, :]).abs().mean().item()
    return g


def main():
    X = load_data()
    print(f"data {tuple(X.shape)}")
    net = V.ConvVAE(V.LATENT, V.BASE).to(DEV)
    ck = torch.load(BEST, map_location=DEV, weights_only=False)
    net.load_state_dict(ck["net"])
    net.eval()
    print(f"loaded best @epoch {ck['epoch']} loss {ck['loss']:.2f}")

    # ---- 全量重建 PSNR / BCE ----
    bce_tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, X.size(0), 384):
            x = X[i:i + 384].to(DEV)
            xr, mu, lv = net(x)
            bce = torch.nn.functional.binary_cross_entropy(
                xr, x, reduction="sum").item()
            bce_tot += bce
            n += x.size(0)
    mean_bce = bce_tot / n
    x8 = X[:8].to(DEV)
    with torch.no_grad():
        r8, _, _ = net(x8)
    ps = psnr(r8, x8)
    print(f"[Recon] mean BCE={mean_bce:.2f}  PSNR(x8)={ps:.2f} dB  "
          f"(32x32 下 PSNR>24 视为不错)")

    # ---- A-E 出图（用 best 权重）----
    with torch.no_grad():
        x64 = X[:64].to(DEV)
        rec64, _, _ = net(x64)
        z = torch.randn(64, V.LATENT, device=DEV)
        samp64 = net.decode(z)
        za, zb = torch.randn(1, V.LATENT, device=DEV), torch.randn(1, V.LATENT, device=DEV)
        al = torch.linspace(0, 1, 10, device=DEV).view(-1, 1)
        interp = net.decode(za * (1 - al) + zb * al)
        mu1, _ = net.encode(X[:1].to(DEV))
        mu2, _ = net.encode(X[1:2].to(DEV))
        recomb = net.decode(torch.cat([mu1[:, :V.LATENT // 2],
                                       mu2[:, V.LATENT // 2:]], 1).expand(1, -1))
    import matplotlib.pyplot as plt

    def save_panel(imgs, title, path, nrow):
        fig, ax = plt.subplots(figsize=(nrow * 0.75 + 1,
                                        ((len(imgs) + nrow - 1) // nrow) * 0.75 + 1),
                               dpi=150)
        ax.imshow(V.show_grid(imgs, nrow).permute(1, 2, 0).numpy())
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    save_panel(x64, "A1. 原始猫图 64 张", rf"{OUT}\vae2best_A1_original.png", 8)
    save_panel(rec64, "A2. VAE 重建（best@ep5953）", rf"{OUT}\vae2best_A2_reconstruct.png", 8)
    save_panel(samp64, "B. 潜空间采样 z~N(0,I)（best）", rf"{OUT}\vae2best_B_latent_sample64.png", 8)
    save_panel(torch.rand(64, 3, 32, 32), "C. 像素空间对照采样",
               rf"{OUT}\vae2best_C_pixel_sample64.png", 8)
    save_panel(interp, "D. 流形插值（非采样）", rf"{OUT}\vae2best_D_interpolate10.png", 10)
    save_panel(recomb, "E. 潜变量重组", rf"{OUT}\vae2best_E_recombine.png", 1)

    # ---- 采样"自然度"对照指标 ----
    print("\n[采样对照] (真实猫图参考 | 潜空间采样 | 像素空间噪声)")
    with torch.no_grad():
        s = samp64.cpu()
    pix = torch.rand(64, 3, 32, 32)
    for name, im in [("real x64", x64.cpu()), ("latent B", s), ("pixel C", pix)]:
        ch_mean = im.mean(dim=(0, 2, 3)).numpy()
        ch_std = im.std(dim=(0, 2, 3)).numpy()
        grad = nat_score(im)
        print(f"{name:9s} meanRGB={np.round(ch_mean,3)} stdRGB={np.round(ch_std,3)} "
              f"edge_grad={grad:.3f}")
    print("\nnotes: edge_grad 越大越锐利；真实猫图 grad≈0.05-0.08，"
          "潜采样应明显高于像素噪声且接近真实值；meanRGB 偏离中心意味着出现色彩结构。")


if __name__ == "__main__":
    main()
