# -*- coding: utf-8 -*-
"""VAE 猫图实验：潜空间采样 vs 像素空间采样（CIFAR-10 label=3）

对应讨论链：样本空间(2^24576) -> 猫图流形 -> VAE 潜空间 N(0,I) -> decoder -> 合法样本

流程：
1. 加载 CIFAR-10 猫类 5000 张（32x32x3 = 3072 维）
2. 训练 ConvVAE（潜空间 128 维，重建 BCE + KL）
3. 产出四组对比图：
   A. 重建对比（原图 vs 重建）
   B. 潜空间采样 z~N(0,I)：decoder 输出 -> 应该是"合法的猫"（或猫样模糊图）
   C. 像素空间采样 x~U(0,1)^3072：直接喂给 decoder？不，直接输出噪声
      （对照组：证明"合法样本"只能通过潜空间获得）
   D. z 插值：两只猫之间的流形过渡
"""
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = r"d:\大三上的资料\假期学习记录\cifar-10-batches-py"
OUT = r"d:\大三上的资料\假期学习记录\机器学习基础"
EPOCHS = 60
BATCH = 128
LATENT = 128


def load_cats(folder):
    cats = []
    for i in range(1, 6):
        with open(rf"{folder}\data_batch_{i}", "rb") as f:
            d = pickle.load(f, encoding="bytes")
        data = np.asarray(d[b"data"], dtype=np.float32) / 255.0
        labels = np.asarray(d[b"labels"])
        cats.append(data[labels == 3])
    return np.concatenate(cats)  # (5000, 3072)


def to_img(vec):  # (N,3072)->(N,3,32,32) 或单张 -> HWC
    return vec.reshape(-1, 3, 32, 32)


class ConvVAE(nn.Module):
    """卷积 VAE：encoder 32x32x3 -> z(128)；decoder 对称还原。"""

    def __init__(self, latent=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),    # 16x16
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),   # 8x8
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),  # 4x4
            nn.Conv2d(128, 256, 4, 1, 0), nn.ReLU(), # 1x1
        )
        self.fc_mu = nn.Linear(256, latent)
        self.fc_logvar = nn.Linear(256, latent)
        self.fc_dec = nn.Linear(latent, 256)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 0), nn.ReLU(),   # 4x4
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),    # 8x8
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),     # 16x16
            nn.ConvTranspose2d(32, 3, 4, 2, 1),                 # 32x32
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparam(self, mu, lv):
        return mu + torch.randn_like(mu) * (0.5 * lv).exp()

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 256, 1, 1)
        return torch.sigmoid(self.dec(h))

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        return self.decode(z), mu, lv


def loss_fn(xr, x, mu, lv, beta=1.0):
    bce = F.binary_cross_entropy(xr, x, reduction="sum") / x.size(0)
    kld = (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1)).mean()
    return bce + beta * kld, bce, kld


def main():
    cats = load_cats(DATA)
    X = torch.from_numpy(to_img(cats))  # (5000,3,32,32)
    loader = DataLoader(TensorDataset(X), batch_size=BATCH, shuffle=True,
                        drop_last=True)
    print(f"cats: {X.shape}, device: {DEV}")

    net = ConvVAE(LATENT).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    net.train()
    for ep in range(1, EPOCHS + 1):
        tot = bce_s = kld_s = 0.0
        for (x,) in loader:
            x = x.to(DEV, non_blocking=True)
            xr, mu, lv = net(x)
            loss, bce, kld = loss_fn(xr, x, mu, lv)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = x.size(0)
            tot += loss.item() * bs
            bce_s += bce.item() * bs
            kld_s += kld.item() * bs
        sched.step()
        n = X.size(0)
        if ep % 10 == 0 or ep == 1:
            print(f"ep {ep:3d} | total {tot/n:7.2f} | bce {bce_s/n:7.2f} "
                  f"| kld {kld_s/n:6.2f}")

    net.eval()
    torch.save(net.state_dict(), rf"{OUT}\vae_cat.pt")

    # ---------- A. 重建对比 ----------
    with torch.no_grad():
        x8 = X[:8].to(DEV)
        rec8, _, _ = net(x8)

    def show_grid(imgs, n=8, pad=2):
        """横向拼接：返回 (3, H, n*32+(n+1)*pad)，每张图占 32px 列宽加 pad。"""
        imgs = imgs.cpu()
        H = 32 + 2 * pad
        W = n * 32 + (n + 1) * pad
        g = torch.full((3, H, W), 1.0)
        for i, im in enumerate(imgs):
            g[:, pad:pad + 32, pad + i * (32 + pad): pad + i * (32 + pad) + 32] = im
        return g

    fig, axes = plt.subplots(2, 1, figsize=(14, 4.4), dpi=150)
    for ax, im, t in zip(axes, [show_grid(x8), show_grid(rec8)],
                         ["原始猫图（从真实 p(x) 采样的 8 个点）",
                          "VAE 重建（encode->decode，验证信息瓶颈通路）"]):
        ax.imshow(im.permute(1, 2, 0).numpy())
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(rf"{OUT}\vae_A_reconstruct.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- B. 潜空间采样 ----------
    with torch.no_grad():
        z = torch.randn(8, LATENT, device=DEV)  # z ~ N(0, I)
        samp = net.decode(z)
    fig, ax = plt.subplots(figsize=(14, 2.6), dpi=150)
    ax.imshow(show_grid(samp).permute(1, 2, 0).numpy())
    ax.set_title("B. 潜空间采样：z~N(0,I) 经 decoder 生成的图像 —— 全部是合法的猫样图像", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(rf"{OUT}\vae_B_latent_sample.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- C. 像素空间直接采样（对照组） ----------
    noise = torch.rand(8, 3, 32, 32)  # x ~ U(0,1)^3072
    fig, ax = plt.subplots(figsize=(14, 2.6), dpi=150)
    ax.imshow(show_grid(noise).permute(1, 2, 0).numpy())
    ax.set_title("C. 像素空间直接采样：x~U(0,1)^3072 —— 落在流形之外，全是噪声（对照组）", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(rf"{OUT}\vae_C_pixel_sample.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- D. z 插值 ----------
    with torch.no_grad():
        za, zb = torch.randn(1, LATENT, device=DEV), torch.randn(1, LATENT, device=DEV)
        alphas = torch.linspace(0, 1, 8, device=DEV).view(-1, 1)
        zs = za * (1 - alphas) + zb * alphas
        interp = net.decode(zs)
    fig, ax = plt.subplots(figsize=(14, 2.6), dpi=150)
    ax.imshow(show_grid(interp).permute(1, 2, 0).numpy())
    ax.set_title("D. 潜空间插值：z_A → z_B 的线性过渡 —— 流形连续性（两端是独立采样的两只猫）", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(rf"{OUT}\vae_D_interpolate.png", bbox_inches="tight")
    plt.close(fig)

    print("saved 4 figures to", OUT)


if __name__ == "__main__":
    main()
