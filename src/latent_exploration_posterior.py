# latent_exploration_posterior.py
# Latent-space exploration for your Probabilistic U-Net (POSTERIOR-based)
# - Uses test set
# - Fixed single context (UNet feature map) for decoding
# - Produces Fig.5a-like joint+marginals and Fig.5b-like grids (HR and residual)
# - Includes dtype fixes, residual visualization, stronger latent excursions,
#   and diagnostics to detect latent usage.

import os
import pickle
import random
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# your modules
import climex_utils as cu
import train_prob_unet_model as tm
from prob_unet import ProbabilisticUNet

# ----------------------------
# 0) Repro, I/O, basic config
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def make_outdir(prefix="latent_exploration_posterior"):
    strtime = datetime.now().strftime('%m/%d/%Y%H:%M:%S')
    out = f"./results/{prefix}/" + strtime + "/"
    os.makedirs(out, exist_ok=True)
    return out

WEIGHTS_PATH = "./results/plots/04/04/202512:43:28/probunet_model_lat_dim_16.pth"
LATENT_DIM   = 16
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------
# 1) Plotting helper: joint+marginals
# -----------------------------------
def plot_joint_with_marginals(s1, s2, evr_pair, outpath, bins=80, title_prefix="Latent space (posterior)"):
    fig = plt.figure(figsize=(7.5, 7.5))
    rect_joint  = [0.1, 0.1, 0.65, 0.65]
    rect_right  = [0.78, 0.1, 0.17, 0.65]
    rect_top    = [0.1, 0.78, 0.65, 0.17]

    ax_joint = fig.add_axes(rect_joint)
    ax_right = fig.add_axes(rect_right, sharey=ax_joint)
    ax_top   = fig.add_axes(rect_top,   sharex=ax_joint)

    h = ax_joint.hist2d(s1, s2, bins=bins, cmap="viridis")
    ax_joint.set_xlabel("PC1 score (s₁)")
    ax_joint.set_ylabel("PC2 score (s₂)")
    cb = fig.colorbar(h[3], ax=ax_joint, fraction=0.046, pad=0.04)
    cb.set_label("Counts")

    ax_top.hist(s1, bins=bins, density=False)
    ax_right.hist(s2, bins=bins, density=False, orientation="horizontal")

    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_top.set_ylabel("Count")
    ax_right.set_xlabel("Count")

    fig.suptitle(
        f"{title_prefix} — PC1: {evr_pair[0]*100:.1f}%  |  PC2: {evr_pair[1]*100:.1f}%",
        y=0.98
    )
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

# -----------------------------------
# 2) Grid builders & inverse mapping
# -----------------------------------
def build_decile_grid(scores, n=10):
    deciles = np.linspace(0.05, 0.95, n)
    pc1_qs = np.quantile(scores[:, 0], deciles)
    pc2_qs = np.quantile(scores[:, 1], deciles)
    return pc1_qs, pc2_qs, deciles

def build_sigma_grid(scores, n=10, sigma=3.0):
    mu1, sd1 = scores[:, 0].mean(), scores[:, 0].std()
    mu2, sd2 = scores[:, 1].mean(), scores[:, 1].std()
    pc1_vals = mu1 + np.linspace(-sigma, sigma, n) * max(sd1, 1e-12)
    pc2_vals = mu2 + np.linspace(-sigma, sigma, n) * max(sd2, 1e-12)
    return pc1_vals, pc2_vals

def make_S_grid(pc1_vals, pc2_vals, D):
    S_grid = []
    for s2 in pc2_vals:           # rows
        for s1 in pc1_vals:       # cols
            s = np.zeros((D,), dtype=np.float32)
            s[0] = s1
            s[1] = s2
            S_grid.append(s)
    return np.stack(S_grid, axis=0)  # [K, D]

def invert_scores_to_latent(S_grid, pca, scaler):
    Zstd = pca.inverse_transform(S_grid)                 # standardized latent (float64)
    Z    = Zstd * scaler.scale_ + scaler.mean_           # original latent space
    return Z.astype(np.float32)                          # ensure float32 for Torch

# -----------------------------------
# 3) Decoding helpers (HR & residual)
# -----------------------------------
@torch.no_grad()
def batched_decode_residual(model, Z_grid_np, feat_fixed, batch_size=32):
    RES_list = []
    K = Z_grid_np.shape[0]
    for start in range(0, K, batch_size):
        end = min(K, start + batch_size)
        z_batch = torch.from_numpy(Z_grid_np[start:end]).to(DEVICE).float()  # float32
        feat_rep = feat_fixed.expand(z_batch.shape[0], -1, -1, -1)
        pred_res = model.fcomb(feat_rep, z_batch).cpu()  # [B, C, H, W]
        for k in range(pred_res.shape[0]):
            RES_list.append(pred_res[k:k+1])
    return RES_list

@torch.no_grad()
def batched_decode_hr(model, Z_grid_np, feat_fixed, lrinterp0_cpu, dataset, batch_size=32):
    HR_list = []
    K = Z_grid_np.shape[0]
    for start in range(0, K, batch_size):
        end = min(K, start + batch_size)
        z_batch = torch.from_numpy(Z_grid_np[start:end]).to(DEVICE).float()
        feat_rep = feat_fixed.expand(z_batch.shape[0], -1, -1, -1)
        pred_res = model.fcomb(feat_rep, z_batch)  # [B, C, H, W]
        for k in range(pred_res.shape[0]):
            res_k = pred_res[k:k+1].cpu()
            hr_k  = dataset.residual_to_hr(res_k, lrinterp0_cpu.to(res_k.dtype))
            HR_list.append(hr_k)
    return HR_list

# -----------------------------------
# 4) Plotters (HR & residual grids)
# -----------------------------------
def _normalize_minmax(arr):
    vmin, vmax = arr.min(), arr.max()
    if vmax <= vmin + 1e-12:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)

def plot_field_grid(tile_list, ncols, nrows, outpath,
                    title, xlab, ylab, chan_idx=0, gain=1.0):
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.8*ncols, 1.8*nrows), constrained_layout=True)
    t = 0
    for j in range(nrows):
        for i in range(ncols):
            ax = axes[j, i] if nrows > 1 else axes[i]
            arr = tile_list[t].numpy()[0, chan_idx] * gain
            img = _normalize_minmax(arr)
            ax.imshow(img, origin="lower", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            t += 1
    fig.suptitle(title, y=0.99, fontsize=12)
    fig.text(0.5, 0.02, xlab, ha="center")
    fig.text(0.02, 0.5, ylab, ha="center", rotation="vertical")
    fig.savefig(outpath, dpi=300); plt.close(fig)

def plot_hr_grid(HR_list, pc1_axis, pc2_axis, outpath, pr_idx=0, label="Deciles"):
    n = len(pc1_axis); m = len(pc2_axis)
    plot_field_grid(
        HR_list, n, m, outpath, chan_idx=pr_idx,
        title=f"HR reconstructions across latent space (posterior) — {label}",
        xlab="PC1 (left→right)", ylab="PC2 (bottom→top)", gain=1.0
    )

def plot_residual_grid(RES_list, pc1_axis, pc2_axis, outpath, pr_idx=0, label="Deciles", gain=3.0):
    n = len(pc1_axis); m = len(pc2_axis)
    plot_field_grid(
        RES_list, n, m, outpath, chan_idx=pr_idx,
        title=f"Residual reconstructions across latent space (posterior) — {label}",
        xlab="PC1 (left→right)", ylab="PC2 (bottom→top)", gain=gain
    )

# -----------------------------------
# 5) Diagnostics for latent usage
# -----------------------------------
@torch.no_grad()
def z_sensitivity_probe(model, feat_fixed, scale=5.0, repeats=3):
    diffs = []
    for _ in range(repeats):
        z0 = torch.zeros((1, LATENT_DIM), device=DEVICE)
        z1 = torch.randn((1, LATENT_DIM), device=DEVICE) * scale
        y0 = model.fcomb(feat_fixed, z0)
        y1 = model.fcomb(feat_fixed, z1)
        diffs.append((y1 - y0).abs().mean().item())
    return float(np.mean(diffs))

def z_weight_vs_feat_weight(model):
    conv0 = model.fcomb.layers[0]  # first 1x1 conv
    W = conv0.weight.data          # [out, in, 1, 1]
    feat_ch = conv0.in_channels - model.latent_dim
    w_feat = W[:, :feat_ch].abs().mean().item()
    w_z    = W[:, feat_ch:].abs().mean().item()
    return w_feat, w_z

# -----------------------------------
# 6) Main
# -----------------------------------
def main():
    set_seed(42)
    outdir = make_outdir()

    # Build args & test set (match your training settings)
    args = tm.get_args()
    args.lowres_scale = 16
    args.batch_size = 32
    args.device = DEVICE

    dataset_test = cu.climex2torch(
        datadir=args.datadir,
        years=args.years_test,
        variables=args.variables,
        coords=args.coords,
        lowres_scale=args.lowres_scale,
        type="lrinterp_to_residuals",
        transfo=True,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Recreate model & load weights
    model = ProbabilisticUNet(
        input_channels=len(args.variables),
        num_classes=len(args.variables),
        latent_dim=LATENT_DIM,
        num_filters=[32, 64, 128, 256],
        model_channels=32,
        channel_mult=[1, 2, 4, 8],
        beta_0=0.0, beta_1=0.0, beta_2=0.0,
    ).to(DEVICE)
    ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.eval()

    # Helper to fetch targets from batch (residuals)
    def get_targets(batch):
        if "targets" in batch:
            return batch["targets"]
        # fallback if dataset didn’t expose 'targets'
        return batch["hr"] - batch["lrinterp"]

    # Collect POSTERIOR means for the whole test set
    all_mu = []
    with torch.no_grad():
        for batch in dataloader_test:
            x = batch["inputs"].to(DEVICE)
            y = get_targets(batch).to(DEVICE)          # <-- TARGETS REQUIRED
            dist = model.posterior(x, y)               # <-- posterior, not prior
            mu = dist.base_dist.loc                    # [B, D]
            all_mu.append(mu.detach().cpu().numpy())
    Z = np.concatenate(all_mu, axis=0).astype(np.float32)  # (N, D)
    N, D = Z.shape
    print(f"[INFO] Collected posterior means: Z shape = {Z.shape}")

    # PCA on standardized Z
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z_std  = scaler.fit_transform(Z)
    pca    = PCA(n_components=D, svd_solver="full", random_state=42)
    S      = pca.fit_transform(Z_std)
    evr    = pca.explained_variance_ratio_[:2]
    print(f"[INFO] PC1, PC2 explained variance: {evr[0]:.4f}, {evr[1]:.4f}")

    with open(os.path.join(outdir, "pca_artifacts.pkl"), "wb") as f:
        pickle.dump({"scaler": scaler, "pca": pca}, f)

    # Fig.5a analog
    plot_joint_with_marginals(
        S[:, 0], S[:, 1], evr_pair=evr,
        outpath=os.path.join(outdir, "fig5a_posterior_joint_marginals.png")
    )

    # Fixed context (first test item)
    sample0 = next(iter(dataloader_test))
    x0 = sample0["inputs"][0:1].to(DEVICE)
    y0 = get_targets(sample0)[0:1].to(DEVICE)
    lrinterp0 = sample0["lrinterp"][0:1].clone()  # CPU tensor
    with torch.no_grad():
        feat0 = model.unet(x0)                     # [1, F, H, W]
        # you can also center future grids around this code if desired:
        mu0  = model.posterior(x0, y0).base_dist.loc.cpu().numpy()[0]

    # Diagnostics (same as before; probes fcomb’s sensitivity to arbitrary z)
    sens = z_sensitivity_probe(model, feat0, scale=5.0, repeats=3)
    w_feat, w_z = z_weight_vs_feat_weight(model)
    print(f"[DIAG] Mean |y(z1)-y(z0)|: {sens:.6f}")
    print(f"[DIAG] ||W_feat||_mean: {w_feat:.6e}  ||W_z||_mean: {w_z:.6e}")

    # Channel to visualize
    var_names = args.variables
    try:
        pr_idx = var_names.index("pr")
    except ValueError:
        pr_idx = 0

    # ------- Grid A: decile midpoints (paper-style) -------
    pc1_qs, pc2_qs, deciles = build_decile_grid(S, n=10)
    S_grid_dec = make_S_grid(pc1_qs, pc2_qs, D)
    Z_grid_dec = invert_scores_to_latent(S_grid_dec, pca, scaler)

    RES_tiles_dec = batched_decode_residual(model, Z_grid_dec, feat0, batch_size=32)
    HR_tiles_dec  = batched_decode_hr(model, Z_grid_dec, feat0, lrinterp0, dataset_test, batch_size=32)

    plot_residual_grid(
        RES_tiles_dec, pc1_qs, pc2_qs,
        outpath=os.path.join(outdir, "fig5b_posterior_residual_grid_deciles.png"),
        pr_idx=pr_idx, label="Deciles", gain=3.0
    )
    plot_hr_grid(
        HR_tiles_dec, pc1_qs, pc2_qs,
        outpath=os.path.join(outdir, "fig5b_posterior_hr_grid_deciles.png"),
        pr_idx=pr_idx, label="Deciles"
    )

    # ------- Grid B: ±3σ excursions (stronger test) -------
    pc1_sig, pc2_sig = build_sigma_grid(S, n=10, sigma=3.0)
    S_grid_sig = make_S_grid(pc1_sig, pc2_sig, D)
    Z_grid_sig = invert_scores_to_latent(S_grid_sig, pca, scaler)

    RES_tiles_sig = batched_decode_residual(model, Z_grid_sig, feat0, batch_size=32)
    HR_tiles_sig  = batched_decode_hr(model, Z_grid_sig, feat0, lrinterp0, dataset_test, batch_size=32)

    plot_residual_grid(
        RES_tiles_sig, pc1_sig, pc2_sig,
        outpath=os.path.join(outdir, "fig5b_posterior_residual_grid_pm3sigma.png"),
        pr_idx=pr_idx, label="±3σ", gain=3.0
    )
    plot_hr_grid(
        HR_tiles_sig, pc1_sig, pc2_sig,
        outpath=os.path.join(outdir, "fig5b_posterior_hr_grid_pm3sigma.png"),
        pr_idx=pr_idx, label="±3σ"
    )

    # Summary
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(f"N samples: {N}\n")
        f.write(f"Latent dim: {D}\n")
        f.write(f"PC1 EVR: {evr[0]:.4f}  PC2 EVR: {evr[1]:.4f}\n")
        f.write(f"DIAG mean |y(z1)-y(z0)|: {sens:.6f}\n")
        f.write(f"DIAG ||W_feat||_mean: {w_feat:.6e}  ||W_z||_mean: {w_z:.6e}\n")
        f.write("Saved:\n")
        f.write("  - fig5a_posterior_joint_marginals.png\n")
        f.write("  - fig5b_posterior_residual_grid_deciles.png\n")
        f.write("  - fig5b_posterior_hr_grid_deciles.png\n")
        f.write("  - fig5b_posterior_residual_grid_pm3sigma.png\n")
        f.write("  - fig5b_posterior_hr_grid_pm3sigma.png\n")

    print(f"[DONE] Outputs written to: {outdir}")

if __name__ == "__main__":
    main()
