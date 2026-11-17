# -*- coding: utf-8 -*-
"""i.i.d. 概念演示：从未知分布采样，样本越多直方图越接近真实分布

对应《机器学习》(周志华) 样本空间/分布/采样/独立同分布 概念：
- 真实分布（曲线）：大自然生成西瓜甜度的"隐形规则"，客观存在但不可直接观测
- 每次采样：买一个瓜测一次甜度
- i.i.d.：每次买瓜互不影响（独立），且都来自同一片瓜田（同分布）
- 直方图逼近曲线：手里样本越多，越能反推出整体规律（机器学习可行性的根基）
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(0)

# 真实分布：瓜田甜度 ~ N(10, 1.2)，8~12 度最常见
mu, sigma = 10.0, 1.2
xs = np.linspace(5.5, 14.5, 400)
pdf = np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

ns = [10, 100, 10_000]
titles = [
    "买 10 个瓜：直方图坑坑洼洼，猜不准规律",
    "买 100 个瓜：大致轮廓出来了",
    "买 10000 个瓜：直方图几乎和真实分布重合",
]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150, sharey=True)
for ax, n, t in zip(axes, ns, titles):
    samples = rng.normal(mu, sigma, n)  # i.i.d. 采样
    ax.hist(samples, bins=30, density=True, alpha=0.55,
            color="tab:green", edgecolor="white", label="样本直方图")
    ax.plot(xs, pdf, color="crimson", lw=2.2, label="真实分布（隐形规则）")
    ax.set_title(t, fontsize=12)
    ax.set_xlabel("西瓜甜度", fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(5.5, 14.5)
axes[0].set_ylabel("概率密度", fontsize=11)
axes[0].legend(loc="upper left", fontsize=9)

fig.suptitle(
    "独立同分布采样：每个瓜都按同一条规律独立抽出 —— 样本越多，越能还原真实分布",
    fontsize=13, y=1.03)
fig.tight_layout()

out = r"d:\大三上的资料\假期学习记录\机器学习基础\iid_sampling_demo.png"
fig.savefig(out, bbox_inches="tight")
print("saved:", out)
