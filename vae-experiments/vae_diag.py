# -*- coding: utf-8 -*-
"""诊断 VAE 重建瓶颈。

问题：视觉重建仍模糊。关键判断：
1. oracle PSNR —— 用 encoder 的 mu（不带采样噪声）直接 decode 的重建质量
2. sample PSNR —— 用 reparam 采样 z 后 decode 的重建质量
   两者接近 => 瓶颈在信息容量(模型/encoder/decoder/latent)，加容量或换损失才有效
   mu 明显更好 => 采样噪声(后验方差)在毁细节，降低 KL/β 或增大 latent 有效
3. 每维度 KL 分布，看 latent 是否有大量"死维度"(未使用)
4. 输出是否只是过度平滑的"均值图"(结构在、纹理全丢)
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"d:\大三上的资料\假期学习记录\机器学习基础")
import vae_cat_full as V

OUT = r"d:\大三上的资料\假期学习记录\机器学习基础"
DEV = "cuda"
NETS = {
    "v2_best": rf"{OUT}\vae_cat_v2_best.pt",
    "ft_best": rf"{OUT}\vae_cat_ft_best.pt",
}
BATCH = 384


def psnr(xr, x):
    mse = ((xr - x) ** 2).mean().item()
    return 10 * np.log10(1.0 / mse) if mse > 0 else 99


def main():
    cats = V.load_cats(V.DATA)
    X = torch.from_numpy(V.to_img(cats))[:1024].to(DEV)
    print(f"probe on first {X.size(0)} cats")

    for name, path in NETS.items():
        net = V.ConvVAE(V.LATENT, V.BASE).to(DEV)
        try:
            ck = torch.load(path, map_location=DEV, weights_only=False)
        except FileNotFoundError:
            print(f"\n[{name}] missing {path}")
            continue
        net.load_state_dict(ck["net"])
        net.eval()
        mu_all = []
        with torch.no_grad():
            # 采样方差与 KL 逐维统计
            ms, vs, samples = [], [], []
            for i in range(0, X.size(0), BATCH):
                x = X[i:i + BATCH]
                mu, lv = net.encode(x)
                ms.append(mu); vs.append(lv.exp())
                z = net.reparam(mu, lv)
                samples.append(z)
            mu = torch.cat(ms); var = torch.cat(vs)
            zs = torch.cat(samples)
            kld = (-0.5 * (1 + mu.var(0) - mu.pow(2).mean(0).mean() - var).mean() + 0.5 * (mu.pow(2) + var).mean()).item()
            # 简化：逐维 KL 均值
            kl_dim = 0.5 * (mu.pow(2) + var - 1 - var.log()).sum(1).mean(0)  # per sample total KL? 不对
            # 正确: per-dim KL ~ N(mu,var) vs N(0,1)
            kl_per_dim = 0.5 * (mu.pow(2) + var - 1 - var.log()).mean(0)  # mean over batch
            # 用 mu 重建 oracle
            rec_mu = net.decode(mu)
            rec_s = net.decode(zs)
            # per-image
            ps_mu = psnr(rec_mu, X)
            ps_s = psnr(rec_s, X)
            act = (kl_per_dim > 0.01).float().mean().item()
            print(f"\n[{name}]")
            print(f"  oracle(mu)   PSNR={ps_mu:5.2f} dB")
            print(f"  sample(repar)PSNR={ps_s:5.2f} dB   (Δ={ps_mu-ps_s:+.2f})")
            print(f"  KL total/sample={kl_per_dim.sum():.2f}  有效维度(kl>0.01)={act*V.LATENT:.0f}/{V.LATENT}")
            print(f"  latent mu std 均值={mu.std(0).mean():.4f} | var均值={var.mean():.4f}")
            # 输出图统计：是否过度平滑 —— 邻域梯度
            with torch.no_grad():
                for tag, im in [("orig", X[:8]), ("rec_mu", rec_mu[:8])]:
                    g = (im[:, :, 1:, :] - im[:, :, :-1, :]).abs().mean().item()
                    print(f"    边缘锐度(梯度幅值) {tag:8s}: {g:.4f}")


if __name__ == "__main__":
    main()
