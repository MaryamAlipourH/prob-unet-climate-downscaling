# metrics.py
import torch
import numpy as np
import pysteps as ps

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def crps_over_groundtruth(hr, preds):
    """
    Compute mean CRPS per variable against ground truth.

    Args
    ----
    hr:    (T, 3, H, W)       torch.Tensor or np.ndarray  [real domain]
    preds: (T, M, 3, H, W)    torch.Tensor or np.ndarray  [real domain]

    Returns
    -------
    dict with scalar means per variable:
      {'pr': float, 'tasmin': float, 'tasmax': float}
    dict with per-timestep arrays for std calculation:  # <-- NEW
      {'pr': array, 'tasmin': array, 'tasmax': array}   # <-- NEW
    """
    preds = _to_numpy(preds)
    hr    = _to_numpy(hr)

    T, M, C, H, W = preds.shape
    assert hr.shape == (T, C, H, W)

    crps_vals = {var: [] for var in ["pr", "tasmin", "tasmax"]}
    # variable order: 0=pr, 1=tasmin, 2=tasmax
    for t in range(T):
        obs_t    = hr[t]          # (3, H, W)
        ens_t    = preds[t]       # (M, 3, H, W)
        # pysteps CRPS returns a 2D field (H, W); take spatial mean
        crps_vals["pr"].append(    ps.verification.probscores.CRPS(ens_t[:, 0], obs_t[0]).mean())
        crps_vals["tasmin"].append(ps.verification.probscores.CRPS(ens_t[:, 1], obs_t[1]).mean())
        crps_vals["tasmax"].append(ps.verification.probscores.CRPS(ens_t[:, 2], obs_t[2]).mean())

    # CHANGED: Return both means and per-timestep arrays
    means = {k: float(np.mean(v)) for k, v in crps_vals.items()}
    arrays = {k: np.array(v) for k, v in crps_vals.items()}  # <-- NEW
    return means, arrays  # <-- CHANGED: now returns tuple

def compute_mae(ground_truth: torch.Tensor, predictions: torch.Tensor) -> tuple:
    """
    ground_truth: (T, 3, H, W) in real units
    predictions:  (T, M, 3, H, W) in real units OR (T, 3, H, W) for deterministic baseline
    
    Returns tuple: (mae_means_dict, mae_arrays_dict)
    """
    # Handle both ensemble and deterministic predictions
    if predictions.dim() == 5:  # Ensemble predictions
        pred_mean = predictions.mean(dim=1)  # (T, 3, H, W)
    else:  # Deterministic predictions (baseline)
        pred_mean = predictions  # Already (T, 3, H, W)
    
    mae_means = {}
    mae_arrays = {}  # <-- NEW
    var_names = ['pr', 'tasmin', 'tasmax']
    
    for i, var in enumerate(var_names):
        abs_diff = torch.abs(ground_truth[:, i] - pred_mean[:, i])  # (T, H, W)
        per_timestep_mae = abs_diff.mean(dim=(1,2))  # Average over H,W -> (T,)  # <-- CHANGED
        mae_means[var] = per_timestep_mae.mean().item()  # Same as before
        mae_arrays[var] = per_timestep_mae.numpy()  # <-- NEW
    
    return mae_means, mae_arrays  # <-- CHANGED: now returns tuple