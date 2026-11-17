# -*- coding: utf-8 -*-
"""VAE 猫图实验 · 完整训练版 v2（CIFAR-10 label=3, 5000 张）

v2 升级（充分利用 8G 显存）：
- 模型：base 64->128（参数量约 x4），latent 256
- 训练：2000 epochs，batch 256，lr 2e-4 + warmup + cosine，beta KL 线性 warmup（0->1，前 50 轮）
- AMP 混合精度训练（fp16 加速 + 省显存）
- 断点续训：每 25 轮存 checkpoint；同时单独保存 loss 最低的最优权重 best
- 产物：损失曲线 + A 重建 + B 潜空间采样 64 张（真正的采样实验）
       + C 像素空间对照 + D 插值（流形连续性，明确标注非采样）+ E 重组
"""
import pickle
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = r"d:\大三上的资料\假期学习记录\cifar-10-batches-py"
OUT = r"d:\大三上的资料\假期学习记录\机器学习基础"
CKPT = rf"{OUT}\vae_cat_v2.pt"          # 续训断点
BEST = rf"{OUT}\vae_cat_v2_best.pt"     # 历史最优权重
EPOCHS = int(os.environ.get("VAE_EPOCHS", 6000))
BATCH = 384
LATENT = 512
BASE = 256
WARMUP_BETA = 50      # KL 权重从 0 线性升到 1 的轮数
WARMUP_LR = 10        # lr warmup 轮数
LR = 3e-4
USE_AMP = DEV == "cuda"
# torch 2.3+ 用 torch.amp.GradScaler，旧版回退 torch.cuda.amp
GradScaler = getattr(torch.amp, "GradScaler", torch.cuda.amp.GradScaler)
torch.backends.cudnn.benchmark = True


def swish(x):
    return x * torch.sigmoid(x)


def load_cats(folder):
    cats = []
    for i in range(1, 6):
        with open(rf"{folder}\data_batch_{i}", "rb") as f:
            d = pickle.load(f, encoding="bytes")
        data = np.asarray(d[b"data"], dtype=np.float32) / 255.0
        labels = np.asarray(d[b"labels"])
        cats.append(data[labels == 3])
    return np.concatenate(cats)


def to_img(vec):
    return vec.reshape(-1, 3, 32, 32)


class ConvVAE(nn.Module):
    def __init__(self, latent=256, base=64):
        super().__init__()
        ch = [3, base, base * 2, base * 4, base * 8]  # 3,64,128,256,512
        # encoder: 32->16->8->4->1
        enc = []
        for i in range(4):  # 32->16->8->4->1（最后一层 k4 s1 p0）
            enc += [nn.Conv2d(ch[i], ch[i + 1], 4, 2 if i < 3 else 1,
                              1 if i < 3 else 0),
                    nn.SiLU()]
        self.enc = nn.Sequential(*enc)
        self.feat_dim = ch[4]  # 512（1x1x512）
        self.fc_mu = nn.Linear(self.feat_dim, latent)
        self.fc_lv = nn.Linear(self.feat_dim, latent)
        self.fc_zd = nn.Linear(latent, 512)
        # decoder: 1->4->8->16->32
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 1, 0), nn.SiLU(),  # 4x4
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.SiLU(),  # 8x8
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(),   # 16x16
            nn.ConvTranspose2d(64, 3, 4, 2, 1),                # 32x32
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)
    def reparam(self, mu, lv):
        return mu + torch.randn_like(mu) * (0.5 * lv).exp()

    def decode(self, z):
        return torch.sigmoid(self.dec(self.fc_zd(z).view(-1, 512, 1, 1)))

    def forward(self, x):
        mu, lv = self.encode(x)
        return self.decode(self.reparam(mu, lv)), mu, lv


def loss_fn(xr, x, mu, lv, beta):
    bce = F.binary_cross_entropy(xr, x, reduction="sum") / x.size(0)
    kld = (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1)).mean()
    return bce + beta * kld, bce, kld


def lr_lambda(ep):
    if ep < WARMUP_LR:
        return (ep + 1) / WARMUP_LR
    t = (ep - WARMUP_LR) / max(1, EPOCHS - WARMUP_LR)
    return 0.5 * (1 + np.cos(np.pi * t))


def show_grid(imgs, ncol, pad=2):
    """网格拼接 (N,3,32,32) -> (3,H,W)，nrow=ncol。"""
    imgs = imgs.cpu()
    n = imgs.size(0)
    nrow = ncol
    ncol_ = (n + nrow - 1) // nrow
    H = ncol_ * 32 + (ncol_ + 1) * pad
    W = nrow * 32 + (nrow + 1) * pad
    g = torch.full((3, H, W), 1.0)
    for i, im in enumerate(imgs):
        r, c = divmod(i, nrow)
        g[:, pad + r * 34: pad + r * 34 + 32,
          pad + c * 34: pad + c * 34 + 32] = im
    return g


def main():
    cats = load_cats(DATA)
    X = torch.from_numpy(to_img(cats))
    loader = DataLoader(TensorDataset(X), batch_size=BATCH, shuffle=True,
                        drop_last=True, pin_memory=True)
    print(f"cats {tuple(X.shape)} device {DEV} params "
          f"{sum(p.numel() for p in ConvVAE(LATENT, BASE).parameters()):,}")

    net = ConvVAE(LATENT, BASE).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = GradScaler("cuda", enabled=USE_AMP)
    start_ep, hist, best_loss = 0, [], float("inf")

    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=DEV, weights_only=False)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_ep, hist, best_loss = ck["epoch"], ck["hist"], ck.get("best", float("inf"))
        print(f"resume from epoch {start_ep}", flush=True)

    t0 = time.time()
    for ep in range(start_ep, EPOCHS):
        net.train()
        beta = min(1.0, (ep + 1) / WARMUP_BETA)
        tot = b = k = 0.0
        pbar = tqdm(loader, desc=f"ep {ep+1}/{EPOCHS}", leave=False)
        for (x,) in pbar:
            x = x.to(DEV, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                xr, mu, lv = net(x)
            with torch.amp.autocast("cuda", enabled=False):  # BCE 不支持 autocast，损失用 fp32
                loss, bce, kld = loss_fn(xr.float(), x, mu.float(), lv.float(), beta)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            n = x.size(0)
            tot += loss.item() * n
            b += bce.item() * n
            k += kld.item() * n
            pbar.set_postfix(loss=f"{loss.item():.1f}")
        sched.step()
        N = X.size(0)
        ep_loss = tot / N
        hist.append((ep_loss, b / N, k / N, beta))
        if ep_loss < best_loss:  # 记录历史最优（loss 最低）权重
            best_loss = ep_loss
            torch.save({"net": net.state_dict(), "epoch": ep + 1,
                        "loss": ep_loss, "hist": hist}, BEST)
        if (ep + 1) % 10 == 0 or ep == start_ep:
            eta = (time.time() - t0) / (ep - start_ep + 1) * (EPOCHS - ep - 1) / 60
            t_, b_, k_, bt = hist[-1]
            print(f"ep {ep+1:4d} beta {bt:.2f} | total {t_:7.2f} | bce {b_:7.2f} "
                  f"| kld {k_:6.2f} | best {best_loss:7.2f} | ETA {eta:5.1f} min",
                  flush=True)
        if (ep + 1) % 25 == 0 or (ep + 1) == EPOCHS:
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": ep + 1, "hist": hist, "best": best_loss}, CKPT)

    # ============ 评估与出图 ============
    net.eval()
    with torch.no_grad():
        # A 重建
        x64 = X[:64].to(DEV)
        rec64, _, _ = net(x64)
        # B 潜空间采样 64 张（真正的采样）
        z = torch.randn(64, LATENT, device=DEV)
        samp64 = net.decode(z)
        # D 插值（流形连续性验证，非采样）
        za, zb = torch.randn(1, LATENT, device=DEV), torch.randn(1, LATENT, device=DEV)
        al = torch.linspace(0, 1, 10, device=DEV).view(-1, 1)
        interp = net.decode(za * (1 - al) + zb * al)
        # E 重组：图1和图2的 z 按维度前半/后半交换
        mu1, _ = net.encode(X[:1].to(DEV))
        mu2, _ = net.encode(X[1:2].to(DEV))
        recomb = net.decode(torch.cat([mu1[:, :LATENT // 2],
                                       mu2[:, LATENT // 2:]], 1).expand(1, -1))

    def save_panel(imgs, title, path, nrow):
        fig, ax = plt.subplots(figsize=(nrow * 0.75 + 1,
                                        ((len(imgs) + nrow - 1) // nrow) * 0.75 + 1),
                               dpi=150)
        ax.imshow(show_grid(imgs, nrow).permute(1, 2, 0).numpy())
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    save_panel(x64, "A1. 原始猫图 64 张（真实 p(x) 的样本）", rf"{OUT}\vae2_A1_original.png", 8)
    save_panel(rec64, "A2. VAE 重建（encode→decode 往返）", rf"{OUT}\vae2_A2_reconstruct.png", 8)
    save_panel(samp64, "B. 潜空间采样 64 张：z~N(0,I) 经 decoder —— 每张都是合法猫样图像（真正的采样实验）",
               rf"{OUT}\vae2_B_latent_sample64.png", 8)
    save_panel(torch.rand(64, 3, 32, 32),
               "C. 对照：像素空间直接采样 x~U(0,1)^3072 —— 落在流形外，全是噪声",
               rf"{OUT}\vae2_C_pixel_sample64.png", 8)
    save_panel(interp, "D. 流形连续性验证（非采样）：z_A→z_B 线性插值，两端为独立采样的 z",
               rf"{OUT}\vae2_D_interpolate10.png", 10)
    save_panel(recomb,
               "E. 潜变量重组：图1 前半 z + 图2 后半 z —— 潜空间结构化证据",
               rf"{OUT}\vae2_E_recombine.png", 1)

    # 损失曲线
    h = np.array(hist)
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), dpi=140)
    ax[0, 0].plot(h[:, 0]); ax[0, 0].set_title("total loss")
    ax[0, 1].plot(h[:, 1], color="tab:orange"); ax[0, 1].set_title("reconstruction (BCE)")
    ax[1, 0].plot(h[:, 2], color="tab:green"); ax[1, 0].set_title("KL divergence")
    ax[1, 1].plot(h[:, 3], color="tab:red"); ax[1, 1].set_title("beta (KL warmup)")
    for a in ax.flat:
        a.set_xlabel("epoch"); a.grid(alpha=0.3)
    fig.suptitle(f"VAE 训练曲线（{EPOCHS} epochs, base={BASE}, latent={LATENT}, batch={BATCH}, AMP, beta warmup {WARMUP_BETA}ep）")
    fig.tight_layout()
    fig.savefig(rf"{OUT}\vae2_loss_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("all figures saved")


if __name__ == "__main__":
    main()
