# -*- coding: utf-8 -*-
"""VAE 猫图 · 第二阶段微调（换重建损失 L1+MSE）

背景：BCE 重建（6000 轮，best loss 1723.99 @ep5953）视觉仍模糊。
本阶段从 best 权重起步，把像素重建损失换成 L1+MSE 加权组合再训，重建更锐利。

关键设计：
- 重建 loss 换为逐像素 L1+MSE（L1 保边缘、MSE 抑制噪点）
- 原量纲：BCE sum/图 ≈1681；L1 仅 ~0.04/像素 —— 直接套会导致 KL 失衡
  => KL 改为逐维均值 (kld/512 ≈ 0.08/维)，与重建同量级
  => β=0.5 重建优先（避免 KL 重新主导、保持 z 仍近似 N(0,I)）
- 低 lr (1e-4) + warmup + cosine，2000 轮；每 25 轮断点 + 单独存重建最优 best
- 全网络微调（encoder 也更新以编码更多细节）
"""
import os
import pickle
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import vae_cat_full as V  # 复用 ConvVAE 结构与数据加载

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = r"d:\大三上的资料\假期学习记录\机器学习基础"
CKPT = rf"{OUT}\vae_cat_ft.pt"           # 二阶段续训断点
BEST0 = rf"{OUT}\vae_cat_v2_best.pt"     # 一阶段最优起点
BESTF = rf"{OUT}\vae_cat_ft_best.pt"     # 二阶段重建最优

FT_EPOCHS = int(os.environ.get("VAE_FT_EPOCHS", 2000))
BATCH = 384
LR = 1e-4
FT_BETA = 0.5          # KL(逐维) 权重：重建优先
WARMUP_LR = 5          # lr warmup 轮数
LATENT = V.LATENT
torch.backends.cudnn.benchmark = True


def recon_loss(xr, x):
    """逐像素平均 L1 + MSE。xr,x ∈ [0,1]。"""
    l1 = (xr - x).abs().mean()
    mse = ((xr - x) ** 2).mean()
    return l1 + mse, l1, mse


def kld_loss(mu, lv):
    """总 KL（每样本对 512 维求和，mean over batch）~43 nat。"""
    return (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1)).mean()


def lr_lambda(ep):
    if ep < WARMUP_LR:
        return (ep + 1) / WARMUP_LR
    t = (ep - WARMUP_LR) / max(1, FT_EPOCHS - WARMUP_LR)
    return 0.5 * (1 + np.cos(np.pi * t))


def main():
    cats = V.load_cats(V.DATA)
    X = torch.from_numpy(V.to_img(cats))
    loader = DataLoader(TensorDataset(X), batch_size=BATCH, shuffle=True,
                        drop_last=True, pin_memory=True)
    print(f"cats {tuple(X.shape)} device {DEV}")

    net = V.ConvVAE(LATENT, V.BASE).to(DEV)
    c0 = torch.load(BEST0, map_location=DEV, weights_only=False)
    net.load_state_dict(c0["net"])  # 从一阶段最优起微调
    print(f"init from best@{c0['epoch']} (loss {c0['loss']:.2f})")

    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = V.GradScaler("cuda", enabled=True)
    start_ep, hist, best = 0, [], float("inf")

    if os.path.exists(CKPT):  # 二阶段自身断点续训
        ck = torch.load(CKPT, map_location=DEV, weights_only=False)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_ep, hist, best = ck["epoch"], ck["hist"], ck.get("best", float("inf"))
        print(f"ft resume from epoch {start_ep}", flush=True)

    t0 = time.time()
    for ep in range(start_ep, FT_EPOCHS):
        net.train()
        beta = FT_BETA
        tr = tl = tm = tk = 0.0
        pbar = tqdm(loader, desc=f"ft {ep+1}/{FT_EPOCHS}", leave=False)
        for (x,) in pbar:
            x = x.to(DEV, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=True):
                xr, mu, lv = net(x)
            with torch.amp.autocast("cuda", enabled=False):
                recon, l1, mse = recon_loss(xr.float(), x)
                kld = kld_loss(mu.float(), lv.float())
                kld_dim = kld / LATENT
                loss = recon + beta * kld_dim
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            n = x.size(0)
            tr += loss.item() * n
            tl += l1.item() * n
            tm += mse.item() * n
            tk += kld.item() * n
            pbar.set_postfix(recon=f"{recon.item():.4f}", kld=f"{kld.item():.1f}")
        sched.step()
        N = X.size(0)
        hist.append((tr / N, tl / N, tm / N, tk / N, beta))
        e_l, e1, em, ek = hist[-1][:4]
        if e_l < best:
            best = e_l
            torch.save({"net": net.state_dict(), "epoch": ep + 1,
                        "loss": e_l, "hist": hist}, BESTF)
        if (ep + 1) % 10 == 0 or ep == start_ep:
            eta = (time.time() - t0) / (ep - start_ep + 1) * (FT_EPOCHS - ep - 1) / 60
            print(f"ft ep {ep+1:4d} | recon {e_l:.4f} (l1 {e1:.4f} mse {em:.5f}) "
                  f"| kld {ek:5.1f} | best {best:.4f} | ETA {eta:4.1f} min", flush=True)
        if (ep + 1) % 25 == 0 or (ep + 1) == FT_EPOCHS:
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": ep + 1, "hist": hist, "best": best}, CKPT)

    # ---- 出图：加载重建最优 best ----
    bf = torch.load(BESTF, map_location=DEV, weights_only=False)
    net.load_state_dict(bf["net"])
    net.eval()
    print(f"figures with ft-best@{bf['epoch']} (recon {bf['loss']:.4f})")
    import matplotlib.pyplot as plt
    from vae_cat_v2_eval import psnr

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

    with torch.no_grad():
        x64 = X[:64].to(DEV)
        rec64, _, _ = net(x64)
        z = torch.randn(64, LATENT, device=DEV)
        samp64 = net.decode(z)
        za, zb = torch.randn(1, LATENT, device=DEV), torch.randn(1, LATENT, device=DEV)
        al = torch.linspace(0, 1, 10, device=DEV).view(-1, 1)
        interp = net.decode(za * (1 - al) + zb * al)
        mu1, _ = net.encode(X[:1].to(DEV))
        mu2, _ = net.encode(X[1:2].to(DEV))
        recomb = net.decode(torch.cat([mu1[:, :LATENT // 2],
                                       mu2[:, LATENT // 2:]], 1).expand(1, -1))
    save_panel(x64, "A1. 原始猫图", rf"{OUT}\vae2ft_A1_original.png", 8)
    save_panel(rec64, "A2. 重建（ft L1+MSE best）", rf"{OUT}\vae2ft_A2_reconstruct.png", 8)
    save_panel(samp64, "B. 潜空间采样 64 张", rf"{OUT}\vae2ft_B_latent_sample64.png", 8)
    save_panel(torch.rand(64, 3, 32, 32), "C. 像素对照",
               rf"{OUT}\vae2ft_C_pixel_sample64.png", 8)
    save_panel(interp, "D. 插值", rf"{OUT}\vae2ft_D_interpolate10.png", 10)
    save_panel(recomb, "E. 重组", rf"{OUT}\vae2ft_E_recombine.png", 1)

    # PSNR(全量重建, 抽 8 张) + 损失曲线
    with torch.no_grad():
        x8 = X[:8].to(DEV)
        r8, _, _ = net(x8)
    print(f"[ft] PSNR(x8) = {psnr(r8, x8):.2f} dB")
    h = np.array(hist)
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), dpi=140)
    ax[0, 0].plot(h[:, 0]); ax[0, 0].set_title("recon L1+MSE")
    ax[0, 1].plot(h[:, 1], color="tab:orange"); ax[0, 1].set_title("L1")
    ax[1, 0].plot(h[:, 2], color="tab:green"); ax[1, 0].set_title("MSE")
    ax[1, 1].plot(h[:, 3], color="tab:red"); ax[1, 1].set_title("KL(total)")
    for a in ax.flat:
        a.set_xlabel("ft epoch"); a.grid(alpha=0.3)
    fig.suptitle(f"VAE 二阶段微调曲线 L1+MSE (beta={FT_BETA}, {FT_EPOCHS}ep)")
    fig.tight_layout()
    fig.savefig(rf"{OUT}\vae2ft_loss_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("all ft figures saved")


if __name__ == "__main__":
    main()
