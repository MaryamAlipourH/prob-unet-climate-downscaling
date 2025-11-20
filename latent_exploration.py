# latent_exploration_prior.py
# Latent-space exploration for your Probabilistic U-Net (PRIOR-based)
# - Uses test set
# - Fixed single context (UNet feature map) for decoding
# - Produces Fig.5a-like joint+marginals and Fig.5b-like grids (HR and residual)
# - Includes dtype fixes, residual visualization, stronger latent excursions,
#   and diagnostics to detect latent collapse.

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
# 0) Reproducibility & outdir
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def make_outdir(prefix="latent_exploration"):
    strtime = datetime.now().strftime('%m/%d/%Y%H:%M:%S')
    out = f"./results/{prefix}/" + strtime + "/"
    os.makedirs(out, exist_ok=True)
    return out

WEIGHTS_PATH = "./results/plots/11/07/202512:30:17/probunet_model_lat_dim_16.pth"
LATENT_DIM   = 16
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------
# 1) Plotting helper: joint+marginals
# -----------------------------------
def plot_joint_with_marginals(s1, s2, evr_pair, outpath, bins=80, title_prefix="Latent space (prior)"):
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
def build_decile_grid(scores, n=6):
    deciles = np.linspace(0.05, 0.95, n)
    pc1_qs = np.quantile(scores[:, 0], deciles)
    pc2_qs = np.quantile(scores[:, 1], deciles)
    return pc1_qs, pc2_qs, deciles

def build_sigma_grid(scores, n=6, sigma=3.0):
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

def invert_scores_to_latent(S_grid, pca, scaler, use_pca=True):
    if not use_pca:
        # No transformation needed, S_grid is already in latent space
        return S_grid.astype(np.float32)
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
        title=f"HR reconstructions across latent space (prior) — {label}",
        xlab="PC1 (left→right)", ylab="PC2 (bottom→top)", gain=1.0
    )

def plot_residual_grid(RES_list, pc1_axis, pc2_axis, outpath, pr_idx=0, label="Deciles", gain=3.0):
    n = len(pc1_axis); m = len(pc2_axis)
    plot_field_grid(
        RES_list, n, m, outpath, chan_idx=pr_idx,
        title=f"Residual reconstructions across latent space (prior) — {label}",
        xlab="PC1 (left→right)", ylab="PC2 (bottom→top)", gain=gain
    )

# -----------------------------------
# 5) Diagnostics for latent usage
# -----------------------------------
@torch.no_grad()
def analyze_prior_distribution(model, dataloader, num_samples=100):
    """Check if prior produces diverse samples"""
    all_z_samples = []
    all_z_stds = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
        x = batch['inputs'][0:1].to(DEVICE)
        
        # Sample from prior multiple times
        prior_dist = model.prior(x)
        z_samples = [prior_dist.rsample() for _ in range(50)]
        z_stack = torch.stack(z_samples)
        
        # Measure per-dimension statistics
        z_mean = z_stack.mean(dim=0)
        z_std = z_stack.std(dim=0)
        
        all_z_samples.append(z_mean)
        all_z_stds.append(z_std)
    
    # Aggregate
    z_stds_avg = torch.stack(all_z_stds).mean(dim=0)
    
    print("\n[DIAG] Prior Distribution Analysis:")
    print(f"  Mean std per dimension: {z_stds_avg.mean().item():.6f}")
    print(f"  Min std: {z_stds_avg.min().item():.6f}")
    print(f"  Max std: {z_stds_avg.max().item():.6f}")
    print(f"  Dimensions with std < 0.1: {(z_stds_avg < 0.1).sum().item()}/{len(z_stds_avg)}")
    
    if z_stds_avg.mean() < 0.5:
        print("WARNING: Prior has collapsed to near-deterministic!")
    
    return z_stds_avg

@torch.no_grad()
def test_extreme_latents(model, feat_fixed, lrinterp, dataset, scale_factors=[0, 1, 3, 5, 10], outpath=None):
    """Test if LARGE latent perturbations create visible changes"""
    results = []
    
    for scale in scale_factors:
        if scale == 0:
            z = torch.zeros(1, LATENT_DIM).to(DEVICE)
        else:
            # Random direction with large magnitude
            z = torch.randn(1, LATENT_DIM).to(DEVICE) * scale
        
        residual = model.fcomb(feat_fixed, z)
        hr = dataset.residual_to_hr(residual.cpu(), lrinterp.cpu())
        results.append((scale, hr))
    
    # Plot side-by-side
    fig, axes = plt.subplots(1, len(scale_factors), figsize=(15, 3))
    for i, (scale, hr) in enumerate(results):
        axes[i].imshow(hr[0, 0], cmap='viridis')
        axes[i].set_title(f'Scale: {scale}')
        axes[i].axis('off')
    
    # Use provided outpath or default
    if outpath is None:
        outpath = 'extreme_latent_test.png'
    
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved extreme latent test to: {outpath}")

@torch.no_grad()
def z_sensitivity_probe(model, feat_fixed, x, scale=5.0, repeats=3):
    """
    Tests if latent variable z (sampled from the LEARNED PRIOR) affects outputs.
    
    Args:
        model: The ProbabilisticUNet model
        feat_fixed: Fixed UNet features [1, F, H, W]
        x: Input tensor [1, C, H, W] to compute prior distribution
        scale: Not used when sampling from learned prior, kept for API compatibility
        repeats: Number of trials
    
    Returns:
        Mean absolute difference between outputs from different z samples
    """
    diffs = []
    for _ in range(repeats):
        # Sample from the LEARNED PRIOR distribution
        prior_dist = model.prior(x)
        z0 = prior_dist.rsample()
        z1 = prior_dist.rsample()
        
        # Decode both samples
        y0 = model.fcomb(feat_fixed, z0)
        y1 = model.fcomb(feat_fixed, z1)
        
        # Measure difference
        diffs.append((y1 - y0).abs().mean().item())
    
    return float(np.mean(diffs))

def z_weight_vs_feat_weight(model):
    # In the new Fcomb, the first combination layer is in 'combine'
    # combine[0] is the first Conv2d that takes [unet_output_channels + latent_dim, ...]
    conv0 = model.fcomb.combine[0]  # Conv2d
    W = conv0.weight.data          # [out, in, 1, 1]
    in_ch = W.shape[1]
    feat_ch = conv0.in_channels - model.latent_dim
    w_feat = W[:, :feat_ch].abs().mean().item()
    w_z    = W[:, feat_ch:].abs().mean().item()
    return w_feat, w_z

@torch.no_grad()
def compute_reconstruction_variance_ratio(model, dataloader, feat_fixed, num_samples=100):
    """
    Compute the ratio of variance due to z vs total reconstruction variance.
    This directly tests if z contributes meaningfully to output diversity.
    """
    # Get one input batch
    batch = next(iter(dataloader))
    x = batch['inputs'][0:1].to(DEVICE)
    
    # Generate multiple samples
    reconstructions = []
    for _ in range(num_samples):
        z = torch.randn(1, model.latent_dim, device=DEVICE)
        y = model.fcomb(feat_fixed, z)
        reconstructions.append(y.cpu())
    
    reconstructions = torch.stack(reconstructions)  # [num_samples, 1, C, H, W]
    
    # Compute variance across samples
    var_across_samples = reconstructions.var(dim=0).mean().item()
    mean_magnitude = reconstructions.abs().mean().item()
    
    # Coefficient of variation
    cv = var_across_samples / (mean_magnitude + 1e-8)
    
    return var_across_samples, mean_magnitude, cv

@torch.no_grad()
def ablation_test_unet_vs_latent(model, x, feat_fixed, target, num_samples=50):
    """
    Compare reconstruction quality when:
    1. Using UNet features + z from PRIOR network (normal operation)
    2. Using zero features + z from PRIOR network (latent only)
    3. Using UNet features + z=0 (deterministic, UNet only)
    4. Using UNet features + raw N(0,1) noise (bypass prior network)
    """
    results = {
        'unet_prior_z': [],      # Normal: UNet + learned prior
        'zero_prior_z': [],      # Latent only: learned prior without UNet
        'unet_zero_z': [],       # UNet only: deterministic
        'unet_random_noise': []  # Bypass: UNet + raw noise (your current test)
    }
    
    for _ in range(num_samples):
        # Sample from the learned PRIOR network
        prior_dist = model.prior(x)
        z_prior = prior_dist.rsample()
        
        # Other z variants
        z_zero = torch.zeros(1, model.latent_dim, device=DEVICE)
        z_random_noise = torch.randn(1, model.latent_dim, device=DEVICE)
        feat_zero = torch.zeros_like(feat_fixed)
        
        # 1. Normal operation: UNet + z from prior
        y1 = model.fcomb(feat_fixed, z_prior)
        results['unet_prior_z'].append((y1 - target).abs().mean().item())
        
        # 2. Latent only: zero features + z from prior
        y2 = model.fcomb(feat_zero, z_prior)
        results['zero_prior_z'].append((y2 - target).abs().mean().item())
        
        # 3. UNet only: UNet features + z=0 (deterministic)
        y3 = model.fcomb(feat_fixed, z_zero)
        results['unet_zero_z'].append((y3 - target).abs().mean().item())
        
        # 4. Bypass prior: UNet + raw Gaussian noise
        y4 = model.fcomb(feat_fixed, z_random_noise)
        results['unet_random_noise'].append((y4 - target).abs().mean().item())
    
    return {k: np.mean(v) for k, v in results.items()}

@torch.no_grad()
def ablation_test_unet_vs_latent_corrected(model, x, feat_fixed, lrinterp, hr_target, dataset, num_samples=50):
    """
    CORRECTED: Compare reconstruction quality in HR space (not residual space).
    """
    results = {
        'unet_prior_z': [],
        'zero_prior_z': [],
        'unet_zero_z': [],
        'unet_random_noise': [],
        'lrinterp_only': []  # NEW: baseline
    }
    
    for _ in range(num_samples):
        # Sample from learned PRIOR network
        prior_dist = model.prior(x)
        z_prior = prior_dist.rsample()
        
        # Other z variants
        z_zero = torch.zeros(1, model.latent_dim, device=DEVICE)
        z_random_noise = torch.randn(1, model.latent_dim, device=DEVICE)
        feat_zero = torch.zeros_like(feat_fixed)
        
        # 1. Normal: UNet + z from prior (then convert to HR)
        residual_1 = model.fcomb(feat_fixed, z_prior).cpu()
        hr_pred_1 = dataset.residual_to_hr(residual_1, lrinterp.cpu())
        results['unet_prior_z'].append((hr_pred_1 - hr_target.cpu()).abs().mean().item())
        
        # 2. Latent only: zero features + z from prior
        residual_2 = model.fcomb(feat_zero, z_prior).cpu()
        hr_pred_2 = dataset.residual_to_hr(residual_2, lrinterp.cpu())
        results['zero_prior_z'].append((hr_pred_2 - hr_target.cpu()).abs().mean().item())
        
        # 3. UNet only: deterministic
        residual_3 = model.fcomb(feat_fixed, z_zero).cpu()
        hr_pred_3 = dataset.residual_to_hr(residual_3, lrinterp.cpu())
        results['unet_zero_z'].append((hr_pred_3 - hr_target.cpu()).abs().mean().item())
        
        # 4. Bypass prior: UNet + raw noise
        residual_4 = model.fcomb(feat_fixed, z_random_noise).cpu()
        hr_pred_4 = dataset.residual_to_hr(residual_4, lrinterp.cpu())
        results['unet_random_noise'].append((hr_pred_4 - hr_target.cpu()).abs().mean().item())
        
        # 5. Baseline: just lrinterp
        results['lrinterp_only'].append((lrinterp.cpu() - hr_target.cpu()).abs().mean().item())
    
    return {k: np.mean(v) for k, v in results.items()}

@torch.no_grad()
def check_output_statistics(model, feat_fixed, target, num_samples=100):
    """
    Check if model outputs are collapsed to near-zero by comparing
    output statistics to target statistics.
    
    This diagnostic reveals whether the model is predicting constant
    near-zero values regardless of the latent variable z.
    """
    outputs = []
    for _ in range(num_samples):
        z = torch.randn(1, model.latent_dim, device=DEVICE)
        y = model.fcomb(feat_fixed, z)
        outputs.append(y)
    
    outputs = torch.cat(outputs, dim=0)  # [num_samples, C, H, W]
    
    target_stats = {
        'mean': target.mean().item(),
        'std': target.std().item(),
        'abs_mean': target.abs().mean().item(),
        'min': target.min().item(),
        'max': target.max().item()
    }
    
    output_stats = {
        'mean': outputs.mean().item(),
        'std': outputs.std().item(),
        'abs_mean': outputs.abs().mean().item(),
        'min': outputs.min().item(),
        'max': outputs.max().item()
    }
    
    # Also compute per-sample variance to see if outputs vary across samples
    output_variance_across_samples = outputs.var(dim=0).mean().item()
    
    return target_stats, output_stats, output_variance_across_samples

def gradient_magnitude_ratio(model, x, target):
    """
    Compute gradient magnitudes w.r.t. z vs w.r.t. UNet features
    to see which pathway has more influence.
    """
    was_training = model.training
    model.train()  # Need gradients
    
    unet_features = model.unet(x)
    unet_features.requires_grad_(True)
    
    # Sample z from posterior
    posterior_dist = model.posterior(x, target)
    z = posterior_dist.rsample()
    z.requires_grad_(True)
    
    # Forward through fcomb
    output = model.fcomb(unet_features, z)
    loss = (output - target).pow(2).mean()
    
    # Compute gradients
    grad_z = torch.autograd.grad(loss, z, retain_graph=True)[0]
    grad_feat = torch.autograd.grad(loss, unet_features)[0]
    
    # Restore original training state
    if not was_training:
        model.eval()
    
    return {
        'grad_z_norm': grad_z.norm().item(),
        'grad_feat_norm': grad_feat.norm().item(),
        'ratio': grad_z.norm().item() / (grad_feat.norm().item() + 1e-8)
    }
@torch.no_grad()
def debug_fcomb_scales(model, feat_fixed, x):
    """
    Debug the Fcomb layer to understand scale mismatches between
    UNet features and latent variables.
    
    Args:
        model: The ProbabilisticUNet model
        feat_fixed: Fixed UNet features [1, F, H, W]
        x: Input tensor [1, C, H, W] to compute prior distribution
    
    Returns:
        Dictionary with magnitude information at each step
    """
    prior_dist = model.prior(x)
    z = prior_dist.rsample()
    
    # Get intermediate outputs
    # Before combining
    feat_magnitude = feat_fixed.abs().mean().item()
    z_magnitude = z.abs().mean().item()
    
    # After tiling z to match spatial dimensions
    H, W = feat_fixed.shape[2], feat_fixed.shape[3]
    
    # Manual tiling to match fcomb's logic
    if z.dim() == 2:
        z_tiled = z.unsqueeze(-1).unsqueeze(-1)
    elif z.dim() == 3:
        z_tiled = z.unsqueeze(-1)
    
    # Tile along height and width
    z_tiled = model.fcomb.tile(z_tiled, dim=2, n_tile=H)
    z_tiled = model.fcomb.tile(z_tiled, dim=3, n_tile=W)
    
    z_tiled_magnitude = z_tiled.abs().mean().item()
    
    # Process latent through its pathway
    z_processed = model.fcomb.latent_processor(z_tiled)
    z_processed_magnitude = z_processed.abs().mean().item()
    
    # Concatenated features
    combined = torch.cat([feat_fixed, z_processed], dim=1)
    combined_magnitude = combined.abs().mean().item()
    
    # Final output
    output = model.fcomb(feat_fixed, z)
    output_magnitude = output.abs().mean().item()
    
    # Check weight magnitudes in first conv layer of combine
    conv0 = model.fcomb.combine[0]  # First Conv2d in combine Sequential
    W = conv0.weight.data
    # The UNet output channels should match feat_fixed.shape[1]
    feat_ch = feat_fixed.shape[1]  # Get from the actual feature map
    w_feat_magnitude = W[:, :feat_ch].abs().mean().item()
    w_z_magnitude = W[:, feat_ch:].abs().mean().item()
    
    # Check if there's a bias term
    bias_magnitude = conv0.bias.abs().mean().item() if conv0.bias is not None else 0.0
    
    # Also check latent_processor weights
    latent_proc_conv0 = model.fcomb.latent_processor[0]
    w_latent_proc = latent_proc_conv0.weight.data.abs().mean().item()
    
    results = {
        'unet_features_magnitude': feat_magnitude,
        'latent_z_magnitude': z_magnitude,
        'z_tiled_magnitude': z_tiled_magnitude,
        'z_processed_magnitude': z_processed_magnitude,
        'combined_magnitude': combined_magnitude,
        'final_output_magnitude': output_magnitude,
        'weights_feat_magnitude': w_feat_magnitude,
        'weights_z_magnitude': w_z_magnitude,
        'weights_latent_processor': w_latent_proc,
        'bias_magnitude': bias_magnitude,
        'z_to_feat_ratio': z_magnitude / (feat_magnitude + 1e-8),
        'w_z_to_w_feat_ratio': w_z_magnitude / (w_feat_magnitude + 1e-8),
        'output_to_input_ratio': output_magnitude / (combined_magnitude + 1e-8)
    }
    
    return results

@torch.no_grad()
def diagnose_latent_pathway_scaling(model, dataloader, num_samples=10):
    """
    Comprehensive analysis of scaling issues in the latent pathway.
    Tests multiple samples to get statistics.
    """
    all_results = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
            
        x = batch['inputs'][0:1].to(DEVICE)
        feat = model.unet(x)
        
        result = debug_fcomb_scales(model, feat, x)
        all_results.append(result)
    
    # Compute statistics across samples
    stats = {}
    for key in all_results[0].keys():
        values = [r[key] for r in all_results]
        stats[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    return stats

# -----------------------------------
# 6) Main
# -----------------------------------
def main():
    set_seed(42)
    outdir = make_outdir()

    # Build args & test set 
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

    # Collect PRIOR means for the whole test set
    all_mu = []
    with torch.no_grad():
        for batch in dataloader_test:
            x = batch["inputs"].to(DEVICE)
            dist = model.prior(x)
            mu = dist.base_dist.loc              # [B, D]
            all_mu.append(mu.detach().cpu().numpy())
    Z = np.concatenate(all_mu, axis=0).astype(np.float32)  # (N, D)
    N, D = Z.shape
    print(f"[INFO] Collected prior means: Z shape = {Z.shape}")

        # After loading model, before PCA
    print("\n" + "="*60)
    print("[DIAG] LATENT SPACE COLLAPSE CHECK")
    print("="*60)

    # Test 1: Prior statistics
    prior_stds = analyze_prior_distribution(model, dataloader_test, num_samples=50)

    # Test 2: Extreme perturbations
    sample0 = next(iter(dataloader_test))
    x0 = sample0["inputs"][0:1].to(DEVICE)
    feat0 = model.unet(x0)
    lrinterp0 = sample0["lrinterp"][0:1]
    test_extreme_latents(
        model, feat0, lrinterp0, dataset_test, 
        scale_factors=[0, 2, 5, 10, 20],
        outpath=os.path.join(outdir, "extreme_latent_test.png")  # ADD THIS
    )

    # Test 3: Quantify visual diversity
    z_samples = []
    outputs = []
    for _ in range(100):
        z = model.prior(x0).rsample()
        out = model.fcomb(feat0, z)
        z_samples.append(z)
        outputs.append(out)

    z_range = torch.stack(z_samples).std(dim=0).mean()
    output_range = torch.stack(outputs).std(dim=0).mean()

    print(f"  Latent z std across samples: {z_range.item():.6f}")
    print(f"  Output std across samples: {output_range.item():.6f}")
    print(f"  Output/Latent amplification: {output_range.item() / (z_range.item() + 1e-8):.2f}x")

    if output_range < 0.05:
        print("  🚨 CRITICAL: Outputs show minimal variation!")

    # Decide whether to apply PCA
    use_pca = (D > 2)
    
    if use_pca:
        # PCA on standardized Z
        scaler = StandardScaler(with_mean=True, with_std=True)
        Z_std  = scaler.fit_transform(Z)
        pca    = PCA(n_components=D, svd_solver="full", random_state=42)
        S      = pca.fit_transform(Z_std)
        evr    = pca.explained_variance_ratio_[:2]
        print(f"[INFO] Applying PCA. PC1, PC2 explained variance: {evr[0]:.4f}, {evr[1]:.4f}")
        
        with open(os.path.join(outdir, "pca_artifacts.pkl"), "wb") as f:
            pickle.dump({"scaler": scaler, "pca": pca}, f)
    else:
        # No PCA: use raw latent space directly
        S = Z  # Already in latent space
        scaler = None
        pca = None
        # For 2D case, compute variance explained by each dimension
        evr = np.var(S, axis=0) / np.var(S).sum()
        print(f"[INFO] Latent dim = 2, skipping PCA. Using raw latent space.")
        print(f"[INFO] Dimension 1 variance ratio: {evr[0]:.4f}, Dimension 2 variance ratio: {evr[1]:.4f}")

    # Fig.5a analog
    title_prefix = "Latent space (prior)" if not use_pca else "Latent space (prior)"
    plot_joint_with_marginals(
        S[:, 0], S[:, 1], evr_pair=evr[:2],
        outpath=os.path.join(outdir, "fig5a_prior_joint_marginals.png"),
        title_prefix=title_prefix
    )

    # Fixed context (first test item)
    sample0 = next(iter(dataloader_test))
    x0 = sample0["inputs"][0:1].to(DEVICE)
    lrinterp0 = sample0["lrinterp"][0:1].clone()  # CPU tensor
    with torch.no_grad():
        feat0 = model.unet(x0)  # [1, F, H, W]

    # Diagnostics
    sens = z_sensitivity_probe(model, feat0, x0, scale=5.0, repeats=3)
    w_feat, w_z = z_weight_vs_feat_weight(model)
    print(f"[DIAG] Mean |y(z1)-y(z0)|: {sens:.6f}")
    print(f"[DIAG] ||W_feat||_mean: {w_feat:.6e}  ||W_z||_mean: {w_z:.6e}")

    # MORE DIAGNOSTICS
    var_z, mean_mag, cv = compute_reconstruction_variance_ratio(model, dataloader_test, feat0, num_samples=100)
    print(f"[DIAG] Reconstruction variance from z: {var_z:.6e}")
    print(f"[DIAG] Mean reconstruction magnitude: {mean_mag:.6e}")
    print(f"[DIAG] Coefficient of variation: {cv:.6f}")

    # ADD THE NEW FCOMB DIAGNOSTICS HERE:
    print("\n" + "="*60)
    print("[DIAG] FCOMB SCALING ANALYSIS")
    print("="*60)
    
    # Single sample analysis
    fcomb_debug = debug_fcomb_scales(model, feat0, x0)
    print(f"  UNet features magnitude: {fcomb_debug['unet_features_magnitude']:.6f}")
    print(f"  Latent z magnitude: {fcomb_debug['latent_z_magnitude']:.6f}")
    print(f"  Tiled z magnitude: {fcomb_debug['z_tiled_magnitude']:.6f}")
    print(f"  Combined magnitude: {fcomb_debug['combined_magnitude']:.6f}")
    print(f"  Final output magnitude: {fcomb_debug['final_output_magnitude']:.6f}")
    print(f"  Weights (feat) magnitude: {fcomb_debug['weights_feat_magnitude']:.6f}")
    print(f"  Weights (z) magnitude: {fcomb_debug['weights_z_magnitude']:.6f}")
    print(f"  Bias magnitude: {fcomb_debug['bias_magnitude']:.6f}")
    print(f"  Z/Feat ratio: {fcomb_debug['z_to_feat_ratio']:.6f}")
    print(f"  W_z/W_feat ratio: {fcomb_debug['w_z_to_w_feat_ratio']:.6f}")
    print(f"  Output/Input ratio: {fcomb_debug['output_to_input_ratio']:.6f}")
    
    # Multi-sample statistics
    print(f"\n  Multi-sample statistics (n=10):")
    scaling_stats = diagnose_latent_pathway_scaling(model, dataloader_test, num_samples=10)
    
    key_metrics = ['z_to_feat_ratio', 'w_z_to_w_feat_ratio', 'output_to_input_ratio', 'final_output_magnitude']
    for metric in key_metrics:
        stats = scaling_stats[metric]
        print(f"    {metric}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, range=[{stats['min']:.6f}, {stats['max']:.6f}]")
    
    # Interpretation
    if fcomb_debug['z_to_feat_ratio'] < 0.01:
        print(f"WARNING: Latent magnitude << UNet magnitude (ratio={fcomb_debug['z_to_feat_ratio']:.6f})")
    if fcomb_debug['w_z_to_w_feat_ratio'] < 0.1:
        print(f"WARNING: Latent weights << UNet weights (ratio={fcomb_debug['w_z_to_w_feat_ratio']:.6f})")
    if fcomb_debug['output_to_input_ratio'] < 0.1:
        print(f"WARNING: Output much smaller than input (ratio={fcomb_debug['output_to_input_ratio']:.6f})")

    # Get a target for ablation
    sample0_target = sample0["hr"][0:1].to(DEVICE)
    
    # NEW: Check output statistics
    target_stats, output_stats, output_var = check_output_statistics(
        model, feat0, sample0_target, num_samples=100
    )
    print(f"[DIAG] Target statistics (ground truth):")
    print(f"  Mean: {target_stats['mean']:.6f}")
    print(f"  Std: {target_stats['std']:.6f}")
    print(f"  Mean(|target|): {target_stats['abs_mean']:.6f}")
    print(f"  Range: [{target_stats['min']:.6f}, {target_stats['max']:.6f}]")
    print(f"[DIAG] Output statistics (100 samples with random z):")
    print(f"  Mean: {output_stats['mean']:.6f}")
    print(f"  Std: {output_stats['std']:.6f}")
    print(f"  Mean(|output|): {output_stats['abs_mean']:.6f}")
    print(f"  Range: [{output_stats['min']:.6f}, {output_stats['max']:.6f}]")
    print(f"  Variance across samples: {output_var:.6e}")
    
    # Interpretation helper
    if output_stats['abs_mean'] < 0.1 * target_stats['abs_mean']:
        print(f" WARNING: Outputs are ~{output_stats['abs_mean']/target_stats['abs_mean']*100:.1f}% of target magnitude - likely predicting near-zero!")
    
    
    # Get data for corrected ablation
    sample0_hr = sample0["hr"][0:1]  # HR ground truth
    lrinterp0_gpu = sample0["lrinterp"][0:1].to(DEVICE)
    
    
    # CORRECTED ablation test
    ablation_correct = ablation_test_unet_vs_latent_corrected(
        model, x0, feat0, lrinterp0_gpu, sample0_hr, dataset_test, num_samples=50
    )
    print(f"\n[DIAG] CORRECTED Ablation test (MAE in HR space):")
    print(f"  Baseline (lrinterp only): {ablation_correct['lrinterp_only']:.4f}")
    print(f"  UNet + prior z: {ablation_correct['unet_prior_z']:.4f}")
    print(f"  Zero + prior z: {ablation_correct['zero_prior_z']:.4f}")
    print(f"  UNet + zero z: {ablation_correct['unet_zero_z']:.4f}")
    print(f"  UNet + raw noise: {ablation_correct['unet_random_noise']:.4f}")
    improvement_pct_ablation = 100*(1 - ablation_correct['unet_prior_z']/ablation_correct['lrinterp_only'])
    print(f"  Improvement over baseline: {improvement_pct_ablation:.2f}%")
    
    grad_info = gradient_magnitude_ratio(model, x0, sample0_target)
    print(f"[DIAG] Gradient magnitudes:")
    print(f"  ||∇_z L||: {grad_info['grad_z_norm']:.6e}")
    print(f"  ||∇_feat L||: {grad_info['grad_feat_norm']:.6e}")
    print(f"  Ratio: {grad_info['ratio']:.6f}")

    # NEW: Analyze residual contribution
    print("\n" + "="*60)
    print("[DIAG] RESIDUAL CONTRIBUTION ANALYSIS")
    print("="*60)
    
    residual_mags = []
    lrinterp_errors = []
    model_errors = []
    
    for i, batch in enumerate(dataloader_test):
        if i >= 20:  # Test on 20 samples
            break
            
        inputs = batch['inputs'].to(DEVICE)
        hr = batch['hr'].to(DEVICE)
        lrinterp = batch['lrinterp'].to(DEVICE)
        timestamps = batch['timestamps'].unsqueeze(1).to(DEVICE)
        
        # Model prediction
        pred_residual = model(inputs, t=timestamps, training=False)
        
        # Reconstruct HR
        hr_pred = dataset_test.residual_to_hr(
            pred_residual.cpu(), 
            lrinterp.cpu()
        ).to(DEVICE)
        
        # Metrics
        residual_mags.append(pred_residual.abs().mean().item())
        lrinterp_errors.append((hr - lrinterp).abs().mean().item())
        model_errors.append((hr - hr_pred).abs().mean().item())
    
    mean_res_mag = np.mean(residual_mags)
    mean_lrinterp_err = np.mean(lrinterp_errors)
    mean_model_err = np.mean(model_errors)
    improvement = mean_lrinterp_err - mean_model_err
    improvement_pct = 100 * (1 - mean_model_err / mean_lrinterp_err)
    
    print(f"  Mean |predicted_residual|: {mean_res_mag:.4f}")
    print(f"  Mean |true_residual|: {mean_lrinterp_err:.4f}")
    print(f"  Error (lrinterp only): {mean_lrinterp_err:.4f}")
    print(f"  Error (lrinterp + model): {mean_model_err:.4f}")
    print(f"  Improvement: {improvement:.4f}")
    print(f"  Improvement %: {improvement_pct:.2f}%")
    
    if improvement_pct < 1.0:
        print(f"  ⚠️  WARNING: Model provides < 1% improvement over bilinear interpolation!")

    # Channel to visualize
    var_names = args.variables
    try:
        pr_idx = var_names.index("pr")
    except ValueError:
        pr_idx = 0

    # ------- Grid A: decile midpoints (paper-style) -------
    pc1_qs, pc2_qs, deciles = build_decile_grid(S, n=6)
    S_grid_dec = make_S_grid(pc1_qs, pc2_qs, D)
    Z_grid_dec = invert_scores_to_latent(S_grid_dec, pca, scaler, use_pca=use_pca)

    RES_tiles_dec = batched_decode_residual(model, Z_grid_dec, feat0, batch_size=32)
    HR_tiles_dec  = batched_decode_hr(model, Z_grid_dec, feat0, lrinterp0, dataset_test, batch_size=32)

    plot_residual_grid(
        RES_tiles_dec, pc1_qs, pc2_qs,
        outpath=os.path.join(outdir, "fig5b_prior_residual_grid_deciles.png"),
        pr_idx=pr_idx, label="Deciles", gain=3.0
    )
    plot_hr_grid(
        HR_tiles_dec, pc1_qs, pc2_qs,
        outpath=os.path.join(outdir, "fig5b_prior_hr_grid_deciles.png"),
        pr_idx=pr_idx, label="Deciles"
    )

    # ------- Grid B: ±3σ excursions (stronger test) -------
    pc1_sig, pc2_sig = build_sigma_grid(S, n=6, sigma=3.0)
    S_grid_sig = make_S_grid(pc1_sig, pc2_sig, D)
    Z_grid_sig = invert_scores_to_latent(S_grid_sig, pca, scaler, use_pca=use_pca)

    RES_tiles_sig = batched_decode_residual(model, Z_grid_sig, feat0, batch_size=32)
    HR_tiles_sig  = batched_decode_hr(model, Z_grid_sig, feat0, lrinterp0, dataset_test, batch_size=32)

    plot_residual_grid(
        RES_tiles_sig, pc1_sig, pc2_sig,
        outpath=os.path.join(outdir, "fig5b_prior_residual_grid_pm3sigma.png"),
        pr_idx=pr_idx, label="±3σ", gain=3.0
    )
    plot_hr_grid(
        HR_tiles_sig, pc1_sig, pc2_sig,
        outpath=os.path.join(outdir, "fig5b_prior_hr_grid_pm3sigma.png"),
        pr_idx=pr_idx, label="±3σ"
    )

    # Summary
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(f"N samples: {N}\n")
        f.write(f"Latent dim: {D}\n")
        if use_pca:
            f.write(f"PCA applied: Yes\n")
            f.write(f"PC1 EVR: {evr[0]:.4f}  PC2 EVR: {evr[1]:.4f}\n")
        else:
            f.write(f"PCA applied: No (latent dim = 2)\n")
            f.write(f"Dim 1 variance ratio: {evr[0]:.4f}  Dim 2 variance ratio: {evr[1]:.4f}\n")
        
        # Existing diagnostics
        f.write(f"DIAG mean |y(z1)-y(z0)|: {sens:.6f}\n")
        f.write(f"DIAG ||W_feat||_mean: {w_feat:.6e}  ||W_z||_mean: {w_z:.6e}\n")
        f.write(f"DIAG Reconstruction CV: {cv:.6f}\n")
        
        # NEW: Add prior distribution stats
        f.write(f"DIAG Prior mean std: {prior_stds.mean().item():.6f}\n")
        f.write(f"DIAG Prior min std: {prior_stds.min().item():.6f}\n")
        f.write(f"DIAG Prior max std: {prior_stds.max().item():.6f}\n")
        f.write(f"DIAG Prior collapsed dims (std<0.1): {(prior_stds < 0.1).sum().item()}/{len(prior_stds)}\n")
        
        # NEW: Add output diversity metrics
        f.write(f"DIAG Latent z std (across samples): {z_range.item():.6f}\n")
        f.write(f"DIAG Output std (across samples): {output_range.item():.6f}\n")
        f.write(f"DIAG Output/Latent amplification: {output_range.item() / (z_range.item() + 1e-8):.2f}x\n")
        
        # Existing stats
        f.write(f"DIAG Target mean(|.|): {target_stats['abs_mean']:.6f}\n")
        f.write(f"DIAG Output mean(|.|): {output_stats['abs_mean']:.6f}\n")
        f.write(f"DIAG Output variance across samples: {output_var:.6e}\n")
        
        # NEW: Add Fcomb scaling metrics
        f.write(f"DIAG Fcomb Z/Feat ratio: {fcomb_debug['z_to_feat_ratio']:.6f}\n")
        f.write(f"DIAG Fcomb W_z/W_feat ratio: {fcomb_debug['w_z_to_w_feat_ratio']:.6f}\n")
        f.write(f"DIAG Fcomb Output/Input ratio: {fcomb_debug['output_to_input_ratio']:.6f}\n")
        
        # Existing ablation and improvement metrics
        f.write(f"DIAG Ablation (corrected) - Baseline: {ablation_correct['lrinterp_only']:.4f}\n")
        f.write(f"DIAG Ablation (corrected) - UNet+priorZ: {ablation_correct['unet_prior_z']:.4f}\n")
        f.write(f"DIAG Ablation (corrected) - UNet+zeroZ: {ablation_correct['unet_zero_z']:.4f}\n")
        f.write(f"DIAG Ablation (corrected) - Improvement: {improvement_pct_ablation:.2f}%\n")
        f.write(f"DIAG Gradient ratio (z/feat): {grad_info['ratio']:.6f}\n")
        f.write(f"DIAG Residual magnitude: {mean_res_mag:.4f}\n")
        f.write(f"DIAG Improvement over lrinterp: {improvement_pct:.2f}%\n")
        
        f.write("Saved:\n")
        f.write("  - fig5a_prior_joint_marginals.png\n")
        f.write("  - fig5b_prior_residual_grid_deciles.png\n")
        f.write("  - fig5b_prior_hr_grid_deciles.png\n")
        f.write("  - fig5b_prior_residual_grid_pm3sigma.png\n")
        f.write("  - fig5b_prior_hr_grid_pm3sigma.png\n")
        f.write("  - extreme_latent_test.png\n")  # NEW

        print(f"[DONE] Outputs written to: {outdir}")

if __name__ == "__main__":
    main()
