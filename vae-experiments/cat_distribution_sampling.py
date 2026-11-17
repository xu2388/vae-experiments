# -*- coding: utf-8 -*-
"""猫图分布采样演示（CIFAR-10, label=3 是猫）— 布局修正版

1. 真实猫图 = 从"白天猫照片分布" p(x) 采出的点（每个点是 3072 维向量）
2. 均匀噪声 = 从错误分布采出的点（同样 3072 维，但完全不像猫）
3. PCA 投影：猫图挤在极小低维流形上，噪声散开得远得多
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(0)
DATA = r"d:\大三上的资料\假期学习记录\cifar-10-batches-py"


def load_cats(folder):
    cats = []
    for i in range(1, 6):
        with open(rf"{folder}\data_batch_{i}", "rb") as f:
            d = pickle.load(f, encoding="bytes")
        data = np.asarray(d[b"data"], dtype=np.float32)
        labels = np.asarray(d[b"labels"])
        cats.append(data[labels == 3])
    return np.concatenate(cats)


def to_img(vec):
    return vec.reshape(3, 32, 32).transpose(1, 2, 0) / 255.0


cats = load_cats(DATA)
print("cat samples:", cats.shape)

n_show = 4
show_idx = rng.choice(len(cats), n_show, replace=False)
noise_vecs = rng.uniform(0, 255, size=(n_show, 3072))

# PCA：拟合猫图（前2主成分），噪声用同一投影
X = cats[:3000]
mu = X.mean(0)
U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
rat = S**2 / (S**2).sum()
cat2d = (X - mu) @ Vt[:2].T
noise2d = (noise_vecs - mu) @ Vt[:2].T

fig = plt.figure(figsize=(15.5, 6.6), dpi=150)
gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.9],
                      hspace=0.35, wspace=0.3)

# 第 0 行：真实猫图；第 1 行：均匀噪声
for j in range(n_show):
    ax0 = fig.add_subplot(gs[0, j])
    ax0.imshow(to_img(cats[show_idx[j]]))
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_title(f"猫图样本 {j+1}", fontsize=10)

    ax1 = fig.add_subplot(gs[1, j])
    ax1.imshow(to_img(noise_vecs[j]))
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title(f"噪声样本 {j+1}", fontsize=10)

# 行标签（旋转文字放在最左）
fig.text(0.015, 0.72, "从 $p(x)$ 采样（真实分布）", rotation=90,
         va="center", ha="center", fontsize=11, color="tab:green")
fig.text(0.015, 0.30, "从均匀分布采样（错误分布）", rotation=90,
         va="center", ha="center", fontsize=11, color="0.4")

# 右侧 PCA 面板（占最后一列、跨两行）
axp = fig.add_subplot(gs[:, -1])
axp.scatter(noise2d[:, 0], noise2d[:, 1], s=16, c="0.5",
            label="噪声样本（散得很开）", zorder=2)
axp.scatter(cat2d[:, 0], cat2d[:, 1], s=7, c="tab:orange", alpha=0.65,
            label="真实猫图（挤在中间小区域）", zorder=3)
axp.set_title("PCA 投影到 2D（同一投影）\n猫图流形 vs 噪声", fontsize=11)
axp.set_xlabel("主成分 1")
axp.set_ylabel("主成分 2")
axp.legend(fontsize=8.5, loc="upper right")
axp.grid(alpha=0.3)
allx = np.concatenate([cat2d[:, 0], noise2d[:, 0]])
ally = np.concatenate([cat2d[:, 1], noise2d[:, 1]])
axp.set_xlim(allx.min() * 1.05, allx.max() * 1.05)
axp.set_ylim(ally.min() * 1.05, ally.max() * 1.05)

fig.suptitle("采样 = 按分布抽一个高维点：快门按下瞬间，自然按 $p(x)$ 抽出一个 3072 维向量\n"
             f"（前 2 主成分仅占方差 {rat[:2].sum():.0%} —— 猫图的真实结构维度远高于 2，但被规律强约束在极小流形上）",
             fontsize=12, y=1.04)
out = r"d:\大三上的资料\假期学习记录\机器学习基础\cat_distribution_sampling.png"
fig.savefig(out, bbox_inches="tight")
print("saved:", out)
