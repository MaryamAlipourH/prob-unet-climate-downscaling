# latent_explore_single_prior.py
# Single-sample latent sweep using the PRIOR (no PCA).
# - Pick one test sample
# - Get its prior q(z|x): mu, sigma
# - Find top-2 sigma dims
# - Build a 6x6 grid over ±3σ along those 2 dims, fix others at mu
# - Decode with fixed UNet features for that sample
# - Save residual and HR grids (per-panel & global), and a Δ-to-center grid

import os
import random
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

import climex_utils as cu
import train_prob_unet_model as tm
from prob_unet import ProbabilisticUNet

# ---------------- Config ----------------
WEIGHTS_PATH = "./results/plots/03/10/202519:32:00/probunet_model_lat_dim_32.pth"
LATENT_DIM   = 32
GRID_SIZE    = 6        # 6x6 grid
SIGMA_RANGE  = 6      # sweep ±3σ
SAMPLE_INDEX = 0        # pick the first test sample
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------- Utilities ---------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def make_outdir(prefix="latent_single_prior"):
    strtime = datetime.now().strftime('%m/%d/%Y%H:%M:%S')
    out = f"./results/{prefix}/" + strtime + "/"
    os.makedirs(out, exist_ok=True)
    return out

def perpanel_norm(arr):
    mn, mx = arr.min(), arr.max()
    if mx <= mn + 1e-12: return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def global_minmax(tiles, ch):
    vals = [t.numpy()[0, ch] for t in tiles]
    vmin = min(v.min() for v in vals)
    vmax = max(v.max() for v in vals)
    if vmax <= vmin + 1e-12:
        vmax = vmin + 1e-12
    return vmin, vmax

def plot_grid(tiles, ncols, nrows, outpath, ch, title, xlab, ylab,
              per_panel=True, vmin=None, vmax=None, gain=1.0):
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.8*ncols, 1.8*nrows),
                             constrained_layout=True)
    t = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c] if nrows > 1 else axes[c]
            a = tiles[t].numpy()[0, ch] * gain
            img = perpanel_norm(a) if per_panel else (a - vmin) / (vmax - vmin + 1e-12)
            ax.imshow(img, origin="lower", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            t += 1
    fig.suptitle(title, y=0.99, fontsize=12)
    fig.text(0.5, 0.02, xlab, ha="center")
    fig.text(0.02, 0.5, ylab, ha="center", rotation="vertical")
    fig.savefig(outpath, dpi=300); plt.close(fig)

def plot_delta_to_center(tiles, ch, outpath, title, center_rc=None):
    n = int(np.sqrt(len(tiles)))
    assert n * n == len(tiles), "Tiles must form a square grid"
    if center_rc is None:
        center_rc = (n // 2, n // 2)
    center_idx = center_rc[0] * n + center_rc[1]
    base = tiles[center_idx].numpy()[0, ch]
    deltas = []
    for t in tiles:
        arr = t.numpy()[0, ch] - base
        deltas.append(torch.tensor(arr[None, None, ...]))  # [1,1,H,W] for reuse
    plot_grid(deltas, n, n, outpath, 0, title, "axis-1", "axis-2", per_panel=True)

# --------- Decoding helpers ----------
@torch.no_grad()
def decode_residual_batch(model, feat_fixed, z_batch):
    # feat_fixed: [1, F, H, W] → expand to [B, F, H, W]
    feat_rep = feat_fixed.expand(z_batch.shape[0], -1, -1, -1)
    y = model.fcomb(feat_rep, z_batch)  # [B, C, H, W]
    return y

@torch.no_grad()
def decode_hr_tiles(model, feat_fixed, z_list, lrinterp_cpu, dataset):
    # z_list: list of [D]-arrays (float32)
    tiles = []
    for z_np in z_list:
        z = torch.from_numpy(z_np[None, :]).to(DEVICE).float()   # [1, D]
        y_res = decode_residual_batch(model, feat_fixed, z)      # [1, C, H, W]
        y_hr  = dataset.residual_to_hr(y_res.cpu(), lrinterp_cpu.to(y_res.dtype))
        tiles.append(y_hr)
    return tiles

@torch.no_grad()
def decode_residual_tiles(model, feat_fixed, z_list):
    tiles = []
    for z_np in z_list:
        z = torch.from_numpy(z_np[None, :]).to(DEVICE).float()
        y_res = decode_residual_batch(model, feat_fixed, z).cpu()
        tiles.append(y_res)
    return tiles

# -------------- Main flow --------------
def main():
    set_seed(42)
    outdir = make_outdir()

    # Args & data (match your training setup)
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
        transfo=True
    )
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                              shuffle=False, num_workers=0)

    # Model
    model = ProbabilisticUNet(
        input_channels=len(args.variables),
        num_classes=len(args.variables),
        latent_dim=LATENT_DIM,
        num_filters=[32, 64, 128, 256],
        model_channels=32,
        channel_mult=[1, 2, 4, 8],
        beta_0=0.0, beta_1=0.0, beta_2=0.0
    ).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE) if hasattr(torch.load, '__call__') else torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    # Fetch a single test sample deterministically
    # Accumulate until we reach SAMPLE_INDEX
    count = 0
    for batch in loader_test:
        B = batch["inputs"].shape[0]
        if count + B <= SAMPLE_INDEX:
            count += B
            continue
        idx_in_batch = SAMPLE_INDEX - count
        x = batch["inputs"][idx_in_batch:idx_in_batch+1].to(DEVICE)    # [1,C,H,W]
        lrinterp = batch["lrinterp"][idx_in_batch:idx_in_batch+1].clone()  # CPU
        break
    else:
        raise IndexError(f"SAMPLE_INDEX {SAMPLE_INDEX} exceeds test set size")

    # Fixed UNet features for this sample
    with torch.no_grad():
        feat = model.unet(x)  # [1, F, H, W]

    # Prior distribution for this sample
    with torch.no_grad():
        dist = model.prior(x)    # Independent(Normal(mu, sigma), 1)
        mu    = dist.base_dist.loc[0].detach().cpu().numpy()   # [D]
        sigma = dist.base_dist.scale[0].detach().cpu().numpy() # [D]

    # Top-2 sigma dimensions
    order = np.argsort(-sigma)  # descending
    i1, i2 = int(order[0]), int(order[1])
    s1, s2 = float(sigma[i1]), float(sigma[i2])
    print(f"[INFO] Top-2 latent dims by σ: {i1} (σ={s1:.4g}), {i2} (σ={s2:.4g})")

    # Build the 6x6 grid of z's: sweep i1 and i2 over ±3σ, others fixed at mu
    a = np.linspace(-SIGMA_RANGE, SIGMA_RANGE, GRID_SIZE)
    b = np.linspace(-SIGMA_RANGE, SIGMA_RANGE, GRID_SIZE)

    z_list = []
    for yb in b:           # rows
        for xa in a:       # cols
            z = mu.astype(np.float32).copy()
            z[i1] = mu[i1] + xa * s1
            z[i2] = mu[i2] + yb * s2
            z_list.append(z)

    # Decode residual and HR tiles
    RES_tiles = decode_residual_tiles(model, feat, z_list)
    HR_tiles  = decode_hr_tiles(model, feat, z_list, lrinterp, dataset_test)

    # Choose channel to visualize (e.g., 'pr' if present)
    var_names = args.variables
    try:
        ch = var_names.index("pr")
    except ValueError:
        ch = 0

    # Plot residual grids (per-panel & global scaling), and Δ-to-center
    n = GRID_SIZE
    vmin, vmax = global_minmax(RES_tiles, ch)
    plot_grid(RES_tiles, n, n,
              os.path.join(outdir, "residual_grid_perpanel.png"),
              ch, f"Residual grid (prior) — dims {i1} vs {i2} (per-panel scaling)",
              f"dim {i1} : -{SIGMA_RANGE}σ → +{SIGMA_RANGE}σ",
              f"dim {i2} : -{SIGMA_RANGE}σ → +{SIGMA_RANGE}σ",
              per_panel=True, gain=3.0)
    plot_grid(RES_tiles, n, n,
              os.path.join(outdir, "residual_grid_global.png"),
              ch, f"Residual grid (prior) — dims {i1} vs {i2} (global scaling)",
              f"dim {i1} : -{SIGMA_RANGE}σ → +{SIGMA_RANGE}σ",
              f"dim {i2} : -{SIGMA_RANGE}σ → +{SIGMA_RANGE}σ",
              per_panel=False, vmin=vmin, vmax=vmax, gain=3.0)
    plot_delta_to_center(RES_tiles, ch,
                         os.path.join(outdir, "residual_delta_to_center.png"),
                         "Residual Δ to center tile (per-panel)")

    # Plot HR grids too (per-panel & global)
    vmin_hr, vmax_hr = global_minmax(HR_tiles, ch)
    plot_grid(HR_tiles, n, n,
              os.path.join(outdir, "hr_grid_perpanel.png"),
              ch, f"HR grid (prior) — dims {i1} vs {i2} (per-panel scaling)",
              f"dim {i1} : -{SIGMA_RANGE}σ → +{SIGMA_RANGE}σ",
              f"dim {i2} : -{SIGMA_RANGE}σ → +{SIGMA_RANGE}σ",
              per_panel=True)
    plot_grid(HR_tiles, n, n,
              os.path.join(outdir, "hr_grid_global.png"),
              ch, f"HR grid (prior) — dims {i1} vs {i2} (global scaling)",
              f"dim {i1} : -{SIGMA_RANGE}σ → +{SIGMA_RANGE}σ",
              f"dim {i2} : -{SIGMA_RANGE}σ → +{SIGMA_RANGE}σ",
              per_panel=False, vmin=vmin_hr, vmax=vmax_hr)

    print(f"[DONE] Saved grids to: {outdir}")

if __name__ == "__main__":
    main()
