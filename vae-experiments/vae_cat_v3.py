# -*- coding: utf-8 -*-
"""VAE 猫图 · v3 全新重训（free-bits 反塌缩 + 大 latent）

针对病灶：v1/v2 latent 512 却只有 ~27 维被用（posterior collapse），
oracle(mu) PSNR 卡在 ~25dB，重建必然模糊。

v3 方案：
- latent 512 -> 1024（更多可用信息通道）
- free-bits：每样本 KL 前 FREE=100 nat 免费、超出才罚，解除"省KL"对维度的压制
- 重建损失 L1+MSE 逐像素（沿用微调方向），recon per-image ≈ L1sum+MSEsum
- 从零全新训练，每 50 轮对固定 probe 测 oracle PSNR + 有效维度，验证维度真被激活
- 产物与 v1/v2 同名加 v3 前缀
"""
import os
import pickle
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import vae_cat_full as V  # 复用 ConvVAE / load_cats / show_grid

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = r"d:\大三上的资料\假期学习记录\机器学习基础"
CKPT = rf"{OUT}\vae_cat_v3.pt"
BEST = rf"{OUT}\vae_cat_v3_best.pt"

EPOCHS = int(os.environ.get("VAE_V3_EPOCHS", 6000))
BATCH = 384
LATENT = 1024
BASE = 256
LR = 3e-4
WARMUP_LR = 10
FREE = 100.0          # free-bits: 每样本 KL 免费额度(nat)，超出才惩罚
BETA = 0.5            # KL 惩罚权重（重建优先）
PIX = 32 * 32 * 3     # 逐图像素总数，用于把 per-pixel 换算成 per-image
torch.backends.cudnn.benchmark = True
# 固定 probe 集（训练集前 512 张），用于监控 oracle PSNR / 有效维度
PROBE = 512


def l1mse_per_img(xr, x):
    """per-image L1+MSE 和。xr,x ∈[0,1], (B,3,32,32)。"""
    d = (xr - x).abs()
    l1 = d.reshape(d.size(0), -1).sum(1).mean()          # ~160
    mse = (d ** 2).reshape(d.size(0), -1).sum(1).mean()  # ~15
    return l1 + mse, l1, mse


def freebits_kld(mu, lv, free=FREE):
    """per-sample KL 总和的免费额度形式：mean(max(KL-free,0))。
    KL = 0.5*(σ²+μ²-1-lnσ²)，σ²=exp(lv)。"""
    var = lv.exp()
    kl = 0.5 * (var + mu.pow(2) - 1 - lv).sum(1)         # (B,), >=0
    return (kl - free).clamp(min=0).mean(), kl.mean()


def lr_lambda(ep):
    if ep < WARMUP_LR:
        return (ep + 1) / WARMUP_LR
    t = (ep - WARMUP_LR) / max(1, EPOCHS - WARMUP_LR)
    return 0.5 * (1 + np.cos(np.pi * t))


def psnr(xr, x):
    mse = ((xr - x) ** 2).mean().item()
    return 10 * np.log10(1.0 / mse) if mse > 0 else 99


def probe(net, Xp):
    """oracle PSNR + 有效维度(kl_per_dim>0.01)。"""
    net.eval()
    with torch.no_grad():
        mu, lv = net.encode(Xp)
        var = lv.exp()
        kl_per_dim = 0.5 * (mu.pow(2) + var - 1 - lv).mean(0)  # mean over batch
        active = (kl_per_dim > 0.01).sum().item()
        rec = net.decode(mu)
        p = psnr(rec, Xp)
    net.train()
    return p, active, kl_per_dim.sum().item()


def main():
    cats = V.load_cats(V.DATA)
    X = torch.from_numpy(V.to_img(cats))
    loader = DataLoader(TensorDataset(X), batch_size=BATCH, shuffle=True,
                        drop_last=True, pin_memory=True)
    Xp = X[:PROBE].to(DEV)
    print(f"cats {tuple(X.shape)} device {DEV} latent={LATENT} "
          f"params={sum(p.numel() for p in V.ConvVAE(LATENT, BASE).parameters()):,} "
          f"free={FREE} beta={BETA}", flush=True)

    net = V.ConvVAE(LATENT, BASE).to(DEV)  # 全新训练
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = V.GradScaler("cuda", enabled=True)
    start_ep, hist = 0, []

    if os.path.exists(CKPT):  # 断点续训（v3 自己的）
        ck = torch.load(CKPT, map_location=DEV, weights_only=False)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_ep, hist = ck["epoch"], ck["hist"]
        print(f"v3 resume from epoch {start_ep}", flush=True)

    best = float("inf")
    t0 = time.time()
    for ep in range(start_ep, EPOCHS):
        net.train()
        s_loss = s_r = s_l = s_m = s_k = 0.0
        pbar = tqdm(loader, desc=f"v3 {ep+1}/{EPOCHS}", leave=False)
        for (x,) in pbar:
            x = x.to(DEV, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=True):
                xr, mu, lv = net(x)
            with torch.amp.autocast("cuda", enabled=False):
                recon, l1, mse = l1mse_per_img(xr.float(), x)
                kpen, ktot = freebits_kld(mu.float(), lv.float())
                loss = recon + BETA * kpen
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            n = x.size(0)
            s_loss += loss.item() * n
            s_r += recon.item() * n
            s_l += l1.item() * n
            s_m += mse.item() * n
            s_k += ktot.item() * n
            pbar.set_postfix(recon=f"{recon.item():.0f}", kl=f"{ktot.item():.0f}")
        sched.step()
        N = X.size(0)
        L = s_loss / N
        hist.append((L, s_r / N, s_l / N, s_m / N, s_k / N))
        if L < best:
            best = L
            torch.save({"net": net.state_dict(), "epoch": ep + 1,
                        "loss": L, "hist": hist}, BEST)
        # 常规进度 + probe 监控
        tag = f"ep {ep+1:4d} | loss {L:6.1f} (l1 {s_l/N:6.1f} mse {s_m/N:5.2f}) | KL {s_k/N:6.1f} | best {best:6.1f}"
        if (ep + 1) % 50 == 0 or ep == start_ep:
            ps, act, klp = probe(net, Xp)
            eta = (time.time() - t0) / (ep - start_ep + 1) * (EPOCHS - ep - 1) / 60
            tag += (f" || oraclePSNR {ps:5.2f} dB | ACTIVE-DIM {act:4d}/{LATENT} "
                    f"| probeKL {klp:6.1f} | ETA {eta:5.0f}min")
            print(tag, flush=True)
        if (ep + 1) % 25 == 0:
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": ep + 1, "hist": hist}, CKPT)
    print("v3 training done", flush=True)

    # ---- 出图（用 best）----
    import matplotlib.pyplot as plt
    bf = torch.load(BEST, map_location=DEV, weights_only=False)
    net.load_state_dict(bf["net"])
    net.eval()
    print(f"figures with v3-best@{bf['epoch']} (loss {bf['loss']:.2f})")
    from vae_cat_v2_eval import psnr as _ps  # noqa

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
    save_panel(x64, "A1. 原始猫图", rf"{OUT}\vae3_A1_original.png", 8)
    save_panel(rec64, "A2. 重建 (v3 free-bits)", rf"{OUT}\vae3_A2_reconstruct.png", 8)
    save_panel(samp64, "B. 潜空间采样 64", rf"{OUT}\vae3_B_latent_sample64.png", 8)
    save_panel(torch.rand(64, 3, 32, 32), "C. 像素对照",
               rf"{OUT}\vae3_C_pixel_sample64.png", 8)
    save_panel(interp, "D. 插值", rf"{OUT}\vae3_D_interpolate10.png", 10)
    save_panel(recomb, "E. 重组", rf"{OUT}\vae3_E_recombine.png", 1)

    ps, act, _ = probe(net, Xp)
    print(f"[v3] final oracle PSNR {ps:.2f} dB | ACTIVE-DIM {act}/{LATENT}")
    h = np.array(hist)
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), dpi=140)
    ax[0, 0].plot(h[:, 0]); ax[0, 0].set_title("loss (recon+beta*freeKL)")
    ax[0, 1].plot(h[:, 1], color="tab:orange"); ax[0, 1].set_title("recon L1+MSE/img")
    ax[1, 0].plot(h[:, 3], color="tab:green"); ax[1, 0].set_title("MSE/img")
    ax[1, 1].plot(h[:, 4], color="tab:red"); ax[1, 1].set_title("KL total/sample")
    for a in ax.flat:
        a.set_xlabel("epoch"); a.grid(alpha=0.3)
    fig.suptitle(f"VAE v3 (latent={LATENT}, free-bits={FREE}, beta={BETA}, L1+MSE)")
    fig.tight_layout()
    fig.savefig(rf"{OUT}\vae3_loss_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("all v3 figures saved")


if __name__ == "__main__":
    main()
