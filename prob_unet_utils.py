import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import genextreme 
from pytorch_msssim import ms_ssim

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

# For plotting the smoothed training and validation losses
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def plot_losses(train_losses_mae, train_losses_kl, train_losses_kl2, val_losses_mae, val_losses_kl, val_losses_kl2, variables, plotdir):
    """
    Plots the training and validation losses for each variable and saves the figure.
    """
    for var in variables:
        fig, ax = plt.subplots(figsize=(15,10))  # Use plt.subplots instead of plt.subplot

        # Ensure that the loss data is converted to NumPy arrays
        train_mae = np.array(train_losses_mae[var])
        val_mae = np.array(val_losses_mae[var])
        train_kl = np.array([loss.detach().cpu().item() for loss in train_losses_kl[var]])
        val_kl = np.array([loss.detach().cpu().item() for loss in val_losses_kl[var]])
        train_kl2 = np.array([loss.detach().cpu().item() for loss in train_losses_kl2[var]])
        val_kl2 = np.array([loss.detach().cpu().item() for loss in val_losses_kl2[var]])

        # Exclude the first epoch loss values to smooth the plot
        if len(train_mae) > 1:
            train_mae = train_mae[3:]
            val_mae = val_mae[3:]
            train_kl = train_kl[3:]
            val_kl = val_kl[3:]
            train_kl2 = train_kl2[3:]
            val_kl2 = val_kl2[3:]

            # Generate epoch numbers starting from 2
            epochs = np.arange(2, len(train_mae) + 2)  # +2 because we start from epoch 2            
        else:
            epochs = np.arange(1, len(train_mae) + 1)  

        # Compute max for KL values (train + val)
        all_kl_values = np.concatenate((train_kl, val_kl)) if len(train_kl) > 0 else np.array([0])
        kl_max = np.max(all_kl_values) if all_kl_values.size > 0 else 0

        # Compute max for KL2 values (train + val)
        all_kl2_values = np.concatenate((train_kl2, val_kl2)) if len(train_kl2) > 0 else np.array([0])
        kl2_max = np.max(all_kl2_values) if all_kl2_values.size > 0 else 0

        # Normalize KL
        if kl_max == 0:
            # If all KL values are zero, set them to 0.5 for a flat line
            train_kl_norm = np.full_like(train_kl, 0.5)
            val_kl_norm = np.full_like(val_kl, 0.5)
        else:
            train_kl_norm = train_kl / kl_max
            val_kl_norm = val_kl / kl_max
        
        if kl2_max ==0:
            train_kl2_norm = np.full_like(train_kl2, 0.5)
            val_kl2_norm = np.full_like(val_kl2, 0.5)
        else:
            train_kl2_norm = train_kl2 / kl2_max
            val_kl2_norm = val_kl2 / kl2_max

        # Plot MAE losses
        ax.plot(epochs, train_mae, lw=2, color='blue', label='Training MAE')
        ax.plot(epochs, val_mae, lw=2, linestyle='dashed', color='blue', label='Validation MAE')
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('MAE Loss', color='blue', fontsize=14)
        ax.tick_params(axis='y', labelcolor='blue')

        # Create a second y-axis for KL and KL2 loss
        ax1 = ax.twinx()

        # Plot KL losses
        kl_line_train, = ax1.plot(epochs, train_kl_norm, lw=2, color='red', label='Training KL (Normalized)')
        kl_line_val, = ax1.plot(epochs, val_kl_norm, lw=2, linestyle='dashed', color='red', label='Validation KL Normalized')

        # Plot KL2 losses
        kl2_line_train, = ax1.plot(epochs, train_kl2_norm, lw=2, color='green', label='Training KL2 Normalized')
        kl2_line_val, = ax1.plot(epochs, val_kl2_norm, lw=2, linestyle='dashed', color='green', label='Validation KL2 Normalized')

        ax1.set_ylabel('Normalized KL Loss', color='red', fontsize=14)
        ax1.tick_params(axis='y', labelcolor='red')

        # Combine legends from both axes
        lines = [ax.get_lines()[0], ax.get_lines()[1],
                 kl_line_train, kl_line_val,
                 kl2_line_train, kl2_line_val]
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc='upper right')

        ax.set_title(f'Training and Validation Losses for {var}', fontsize=16)
        ax.grid(True)
        fig.tight_layout()


        # plt.show()

        # Save the figure
        fig.savefig(f"{plotdir}/loss_{var}.png", dpi=300)
        plt.close(fig)


def plot_losses_mae(train_losses_mae, val_losses_mae, variables, plotdir):
    """
    Plots the training and validation losses for each variable and saves the figure.
    """
    for var in variables:
        fig, ax = plt.subplots(figsize=(15,10))  # Use plt.subplots instead of plt.subplot

        # Ensure that the loss data is converted to NumPy arrays
        train_mae = np.array(train_losses_mae[var])
        val_mae = np.array(val_losses_mae[var])


        # Generate epoch numbers starting from 2
        epochs = np.arange(1, len(train_mae) + 1)  # +2 because we start from epoch 2

        # Plot MAE losses
        ax.plot(epochs, train_mae, lw=2, color='blue', label='Training MAE')
        ax.plot(epochs, val_mae, lw=2, linestyle='dashed', color='blue', label='Validation MAE')
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('MAE Loss', color='blue', fontsize=14)
        ax.tick_params(axis='y', labelcolor='blue')


        ax.set_title(f'Training and Validation Losses for {var}', fontsize=16)
        ax.grid(True)
        ax.legend(loc='upper right', fontsize=12)  # Ensure legend is visible           
        fig.tight_layout()
        # plt.show()

        # Save the figure
        fig.savefig(f"{plotdir}/loss_{var}.png", dpi=300)
        plt.close(fig)


def compute_annual_block_maxima(daily_data, years, days_per_year=365):
    """
    Block-maxima approach for annual extremes.

    daily_data: shape [N_days, N_realizations]
        daily_data[d, r] = precipitation on day d, realization r.

    years: list or range of the actual years (e.g., 1960..2005). 
           Must match length of daily_data // days_per_year.

    Returns:
        block_maxima: 1D np.array of length (len(years) * N_realizations)
    """
    num_years = len(years)
    block_maxima = []

    for y in range(num_years):
        start_idx = y * days_per_year
        end_idx = (y + 1) * days_per_year
        # slice for this year => shape [days_per_year, N_realizations]
        slice_data = daily_data[start_idx:end_idx, :]
        maxima_this_year = slice_data.max(axis=0)  # shape (N_realizations,)
        block_maxima.extend(maxima_this_year.tolist())

    return np.array(block_maxima)


def gev_return_level(shape, loc, scale, return_period):
    """
    For SciPy's genextreme parameterization:
      shape = kappa, loc = mu, scale = sigma

    return_period: T-year
    Return the T-year return level z_T.
    """
    cdf_val = 1.0 - 1.0 / return_period
    rl = genextreme.ppf(cdf_val, shape, loc=loc, scale=scale)
    return rl


def gev_parametric_bootstrap(shape_hat, loc_hat, scale_hat, sample_size,
                             return_periods=[2,5,10,20,50,100],
                             n_bootstrap=200,
                             random_state=42):
    """
    Parametric bootstrap to get confidence intervals for GEV return levels.

    shape_hat, loc_hat, scale_hat: MLE-fitted GEV parameters (SciPy style).
    sample_size: number of block maxima in the original sample
    return_periods: list of T-year return periods
    n_bootstrap: how many bootstrap resamples
    """
    rng = np.random.default_rng(seed=random_state)
    rl_distributions = {T: [] for T in return_periods}

    for _ in range(n_bootstrap):
        # 1) Generate new sample from the fitted GEV
        synthetic_data = genextreme.rvs(shape_hat, loc=loc_hat, scale=scale_hat,
                                        size=sample_size, random_state=rng)
        # 2) Fit GEV
        shape_b, loc_b, scale_b = genextreme.fit(synthetic_data)

        # 3) Compute RL for each T
        for T in return_periods:
            rl_b = gev_return_level(shape_b, loc_b, scale_b, T)
            rl_distributions[T].append(rl_b)

    return rl_distributions

def get_empirical_return_periods(block_maxima):
    """
    Sort block maxima in descending order and compute empirical return periods
    using T_i = (N+1)/i.

    Args:
        block_maxima (1D array): block maxima data, length N

    Returns:
        sorted_desc (1D array, length N): block maxima sorted descending
        T_empirical (1D array, length N): return periods for each sorted value
    """
    # Sort descending
    sorted_desc = np.sort(block_maxima)[::-1]
    N = len(sorted_desc)
    i_vals = np.arange(1, N+1)
    T_empirical = (N + 1) / i_vals
    return sorted_desc, T_empirical



def afcrps_loss(ensemble_pred: torch.Tensor,
                target: torch.Tensor,
                alpha: float = 0.95) -> torch.Tensor:
    """
    Computes the almost-fair CRPS by summing nonnegative terms for j != k:
        afCRPS_α = 1 / [2 M (M-1)] * Σ_{j≠k} (|x_j - y| + |x_k - y| - (1-ε) |x_j - x_k|)
    where ε = (1-α)/M.

    Args:
        ensemble_pred: Tensor [batch, M, channels, height, width]
        target:        Tensor [batch, channels, height, width] or [batch, M, channels, height, width]
        alpha:         Mixing parameter, close to 1 for stability.

    Returns:
        A scalar (mean over the batch) torch.Tensor with the afCRPS loss.
    """
    B, M, C, H, W = ensemble_pred.shape

    # Expand target to match [B, M, C, H, W] if needed
    if target.dim() == 4:
        target = target.unsqueeze(1)  # now [B,1,C,H,W]
        target = target.expand(-1, M, -1, -1, -1)  # [B,M,C,H,W]

    # Epsilon parameter from α
    eps = (1.0 - alpha) / M

    # We will form the quantity (|x_j - y| + |x_k - y| - (1-ε)|x_j - x_k|) in a pairwise manner.
    # 1) Get |x_j - y| by broadcasting to [B, M, 1, C, H, W] and similarly for k.
    x_minus_y = ensemble_pred - target  # [B,M,C,H,W]
    xj_y = x_minus_y.unsqueeze(2)       # => [B, M, 1, C, H, W]
    xk_y = x_minus_y.unsqueeze(1)       # => [B, 1, M, C, H, W]
    term_jy_ky = torch.abs(xj_y) + torch.abs(xk_y)  # => [B, M, M, C, H, W]

    # 2) Get (1-ε)*|x_j - x_k|
    xj = ensemble_pred.unsqueeze(2)  # [B, M, 1, C, H, W]
    xk = ensemble_pred.unsqueeze(1)  # [B, 1, M, C, H, W]
    term_jk = (1.0 - eps) * torch.abs(xj - xk)  # => [B, M, M, C, H, W]

    # 3) Combine them: (|x_j - y| + |x_k - y| - (1-ε)|x_j - x_k|)
    combined = term_jy_ky - term_jk  # => [B, M, M, C, H, W]

    # 4) Exclude j = k terms by setting them to zero, since formula uses k ≠ j
    #    (Alternatively, we could sum only j < k and multiply by 2, but here's a direct approach.)
    #    Make a [M,M] mask that is 1 if j≠k, 0 if j=k.
    mask = 1.0 - torch.eye(M, dtype=ensemble_pred.dtype, device=ensemble_pred.device)
    # shape [M,M], with 1s off-diagonal, 0 on diagonal

    # Expand mask to shape [1, M, M, 1, 1, 1] so it can broadcast
    mask = mask.view(1, M, M, 1, 1, 1)
    combined = combined * mask  # zero out diagonal terms

    # 5) Sum over j,k,C,H,W => reduce to [B]
    #    sum(dim=(1,2,3,4,5)) sums M, M, channels, height, width
    sum_per_batch = combined.sum(dim=(1,2,3,4,5))  # shape [B]

    # 6) Multiply by the normalizing factor 1 / [2 M (M-1)]
    factor = 1.0 / (2.0 * M * (M - 1))

    pixel_factor = 1.0 / (C * H * W)

    afcrps_per_batch = factor * pixel_factor * sum_per_batch  # [B]

    # 7) Final loss is mean over the batch
    return afcrps_per_batch.mean()


def crps_loss(ensemble_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the CRPS loss for an ensemble.

    Args:
        ensemble_pred (Tensor): [B, M, C, H, W] - ensemble predictions
        target (Tensor): [B, C, H, W] or broadcastable to [B, M, C, H, W]

    Returns:
        Scalar CRPS loss (averaged over batch, channels, spatial dims).
    """
    B, M, C, H, W = ensemble_pred.shape

    # Expand target to shape [B, M, C, H, W] if needed
    if target.dim() == 4:
        target = target.unsqueeze(1).expand(-1, M, -1, -1, -1)

    # First term: mean over ensemble of |x_j - y|
    abs_diff_to_target = torch.abs(ensemble_pred - target)  # [B, M, C, H, W]
    first_term = abs_diff_to_target.mean(dim=1)  # mean over M => [B, C, H, W]

    # Second term: mean over all pairs |x_j - x_k|
    xj = ensemble_pred.unsqueeze(2)  # [B, M, 1, C, H, W]
    xk = ensemble_pred.unsqueeze(1)  # [B, 1, M, C, H, W]
    abs_pairwise_diff = torch.abs(xj - xk)  # [B, M, M, C, H, W]
    second_term = abs_pairwise_diff.mean(dim=(1, 2))  # mean over M^2 => [B, C, H, W]

    # CRPS = first_term - 0.5 * second_term
    crps = first_term - 0.5 * second_term

    # Final loss: average over batch, channels, height, width
    return crps.mean()

def wmse_ms_ssim_loss(pred: torch.Tensor,
                      target: torch.Tensor,
                      alpha: float = 0.007,
                      beta:  float = 0.048,
                      lam:   float = 0.000,
                      data_range: float | None = None) -> torch.Tensor:
    """
    Implements L_λ(y,ŷ) = λ·WMSE + (1-λ)·(1-MS-SSIM)

      WMSE = (1/N) Σ w(y_i)(y_i-ŷ_i)^2 ,   w(y_i) = min(α·e^{β·y_i}, 1)

    * `pred`, `target`:  [B, C, H, W]  (or broadcastable)
    * `data_range`      :  value range of the data; if None it is inferred.
    """
    if pred.dim() == 5:           # we called loss on an ensemble [B,M,C,H,W]
        pred = pred.mean(1)       # → use its mean

    if data_range is None:
        data_range = (target.max() - target.min()).clamp(min=1e-5).item()

    # --- weighted MSE -------------------------------------------------
    weights = torch.clamp(alpha * torch.exp(beta * target), max=1.0)
    wmse = (weights * (pred - target).pow(2)).mean()

    # --- (1-MS-SSIM) --------------------------------------------------
    # ms_ssim already returns a mean over the batch when size_average=True
    msssim_val = ms_ssim(pred, target, data_range=data_range, size_average=True, win_size=7)
    msssim_loss = 1.0 - msssim_val          # lower is better

    # --- convex combination ------------------------------------------
    return lam * wmse + (1.0 - lam) * msssim_loss
