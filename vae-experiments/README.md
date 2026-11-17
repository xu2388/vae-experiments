# VAE 猫脸图像生成实验（v1–v3）

在猫脸图像上训练变分自编码器（VAE），重点研究潜变量坍缩（posterior collapse）与重建质量之间的关系。

## 实验版本

| 脚本 | 版本 |
|------|------|
| `vae_cat_experiment.py` | v1 基线 |
| `vae_cat_full.py` / `vae_cat_ft.py` | 完整训练与微调 |
| `vae_cat_v2.py` / `vae_cat_v2_eval.py` | v2：增大 latent、调整重建损失 |
| `vae_cat_v3.py` | v3：free-bits + latent 1024，缓解 KL 对维度的压制 |

## 结果图

`vae_*_A1_original.png`、`A2_reconstruct`、`B_latent_sample`、`C_pixel_sample`、`D_interpolate`、`E_recombine`、`loss_curves` 等：原图与重建、潜空间采样、像素空间采样、插值、特征重组、损失曲线。

## 怎么跑

```bash
pip install torch torchvision numpy matplotlib
python vae_cat_v3.py
```

## 说明

模型权重（`*.pt`）与训练日志未随仓库上传。
