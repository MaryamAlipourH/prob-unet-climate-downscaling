import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import argparse
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import climex_utils as cu

warnings.filterwarnings('ignore')

def get_args():

    """
    This function returns a dictionary containing all necessary arguments for importing ClimEx data, training, evaluating and sampling from a downscaling ML model.
    This function is helpful for doing sweeps and performing hyperparameter tuning.
    """

    parser = argparse.ArgumentParser()

    # climate dataset arguments
    parser.add_argument('--datadir', type=str, default='/home/julie/Data/Climex/day/kdj/')
    parser.add_argument('--variables', type=list, default=['pr', 'tasmin', 'tasmax'])
    parser.add_argument('--years_train', type=range, default=range(1960, 2020))
    parser.add_argument('--years_val', type=range, default=range(2021, 2033))
    parser.add_argument('--years_test', type=range, default=range(2034, 2046))
    # parser.add_argument('--years_subtrain', type=range, default=range(1960, 1980))
    # parser.add_argument('--years_earlystop', type=range, default=range(1980, 1990))
    
    # parser.add_argument('--years_megatrain', type=range, default=range(1960, 1998))
    
    parser.add_argument('--coords', type=list, default=[80, 208, 100, 228]) 
    parser.add_argument('--resolution', type=tuple, default=(128, 128))
    parser.add_argument('--lowres_scale', type=int, default=16)
    parser.add_argument('--timetransform', type=str, default='id', choices=['id', 'cyclic'])
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs for beta_2 increase')
    parser.add_argument('--loss_type', type=str, default='mse+ssim', choices=['afcrps', 'crps', 'mse+ssim'])


    # Downscaling method 
    parser.add_argument('--ds_model', type=str, default='deterministic_unet', choices=['deterministic_unet', 'probabilistic_unet', 'vae', 'linearcnn', 'bcsd'])

    # ML training arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--accum', type=int, default=8)
    parser.add_argument('--optimizer', type=object, default=torch.optim.AdamW)

    # Model evaluation arguments
    #parser.add_argument('--metric', type=str, default="rmse", choices=["rmse", "crps", "rapsd", "mae"])

    # WandB activation 
    parser.add_argument('--wandb', type=bool, default=False)

    # GPU
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # saving results arguments
    strtime = datetime.now().strftime('%m/%d/%Y%H:%M:%S')
    plotdir = './results/plots/' + strtime + '/'
    parser.add_argument('--plotdir', type=str, default=plotdir)
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    # Generating the dictionary
    args, unknown = parser.parse_known_args()

    return args

class EarlyStopper:

    """
    Class for early stopping in the training loop.
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss, model):

        # if the validation loss is lower than the previous minimum, save the model as best model
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            torch.save(model.state_dict(), f"./last_best_model_hr.pt")
            self.counter = 0

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            # if the counter is greater than the patience, load the best model and return True to break training
            if self.counter >= self.patience:
                model.load_state_dict(torch.load(f"./last_best_model_hr.pt"))
                return True, model
        return False, model

# train_probunet_step is a function that performs one training epoch with afCRPS as reconstruction loss.
def train_probunet_step(model, dataloader, optimizer, epoch, num_epochs, device, ensemble_size=5):
    """
    Performs one training epoch with afCRPS as reconstruction loss.
    
    Returns:
        mean_crps  (float): Average afCRPS over the epoch
        mean_kl    (float): Average KL divergence (posterior vs prior)
    """
    model.train()
    # recon_vals, kl_vals, wmse_vals, msssim_vals = [], [], [], [],
    recon_vals, crps_vals, kl_vals = [], [], []


    with tqdm(total=len(dataloader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{num_epochs}")

        for batch in dataloader:
            tq.update(1)
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            timestamps = batch['timestamps'].unsqueeze(dim=1).to(device)

            # Forward pass: get total ELBO-like loss (with afCRPS),
            # plus the scalar CRPS, KL, KL2 for logging
            # loss, recon_list, kl_div, wmse, msssim = model.elbo(
            #     inputs, targets, timestamps, 
            #     M=ensemble_size 
            # )
            loss, recon_list, kl_div = model.elbo(
                inputs, targets, timestamps, 
                M=ensemble_size 
            )

            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # crps_list is just [crps_value], so record it
            recon_vals.append(recon_list[0])
            kl_vals.append(kl_div.mean().item())
            # wmse_vals.append(wmse)
            # msssim_vals.append(msssim)

            # Show running loss in the progress bar
            tq.set_postfix_str(s=f"Loss: {loss.item():.4f}")

    mean_crps = float(np.mean(recon_vals))
    mean_kl = float(np.mean(kl_vals))
    # mean_wmse = float(np.mean(wmse_vals))
    # mean_msssim = float(np.mean(msssim_vals))

    # return mean_crps, mean_kl, mean_wmse, mean_msssim
    return mean_crps, mean_kl

# eval_probunet_model is a function that evaluates the Probabilistic U-Net model on a validation/test dataset using afCRPS (no per-variable MAE).
@torch.no_grad()
def eval_probunet_model(model, dataloader, device, ensemble_size=5):
    """
    Evaluates the Probabilistic U-Net model on a validation/test dataset
    using afCRPS (no per-variable MAE).
    
    Returns:
        mean_crps  (float): Average afCRPS across the dataset
        mean_kl    (float): Average KL divergence (posterior vs prior)
    """
    model.eval()
    crps_vals = []
    kl_vals = []
    wmse_vals = []
    msssim_vals = []

    with tqdm(total=len(dataloader), dynamic_ncols=True) as tq:
        tq.set_description(":: Evaluation ::")

        for batch in dataloader:
            tq.update(1)
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            timestamps = batch['timestamps'].unsqueeze(dim=1).to(device)

            # # Call your ELBO with multiple ensemble samples => returns afCRPS
            # loss, crps_list, kl_div, wmse, msssim = model.elbo(
            #     inputs, targets, timestamps, 
            #     M=ensemble_size
            # )

                        # Call your ELBO with multiple ensemble samples => returns afCRPS
            loss, crps_list, kl_div = model.elbo(
                inputs, targets, timestamps, 
                M=ensemble_size
            )

            # crps_list is just [afCRPS_value], so we take crps_list[0].
            crps_vals.append(crps_list[0])
            kl_vals.append(kl_div.mean().item())
            # wmse_vals.append(wmse)
            # msssim_vals.append(msssim)

    mean_crps = float(np.mean(crps_vals))
    mean_kl = float(np.mean(kl_vals))
    # mean_wmse = float(np.mean(wmse_vals))
    # mean_msssim = float(np.mean(msssim_vals))

    # return mean_crps, mean_kl, mean_wmse, mean_msssim
    return mean_crps, mean_kl


@torch.no_grad()
def sample_probunet_model(model, dataloader, epoch, device, batch=None):

    """
    Generates and plots samples from the Probabilistic U-Net model.

    Args:
        model (ProbabilisticUNet): The trained Probabilistic U-Net model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        epoch (int): Current epoch number (used for labeling plots).
        device (torch.device): Computation device (CPU or GPU).
        batch: if batch is provided it uses the batch directly. If not, it takes a random batch from the dataloader.

    Returns:
        torch.Tensor: The generated high-resolution predictions.
    """
    model.eval()
    if batch is None:
        batch = next(iter(dataloader))
        
    inputs = batch['inputs'][:2].to(device)  # Select 2 random low-resolution inputs
    lrinterp = batch['lrinterp'][:2].to(device)
    hr = batch['hr'][:2].to(device)
    # Ensure timestamps has shape (batch_size, 1)
    timestamps = batch['timestamps'][:2].unsqueeze(dim=1).to(device)
    timestamps_float = batch['timestamps_float'][:2]  # For plotting
   

    num_samples = 3  # Number of predicted high-resolution outputs per input
    hr_preds = []

    for _ in range(num_samples):
        output = model(inputs, t=timestamps, training=False) # Generate output from the model
        hr_pred = dataloader.dataset.residual_to_hr(output.cpu(), lrinterp.cpu())  # Convert residual to high-res
        hr_preds.append(hr_pred) # Append the prediction to the list


    # Stack the predictions along a new dimension to create a tensor of shape [batch_size, num_samples, channels, height, width]
    hr_preds = torch.stack(hr_preds, dim=1)  # Shape: [batch_size, num_samples, channels, height, width]

    # Plot the generated samples using the custom plotting function
    fig, axs = dataloader.dataset.plot_sample_batch(lrinterp.cpu(), hr_preds.cpu(), hr.cpu(), timestamps_float, epoch, N=2, num_samples=num_samples)

    return hr_preds, (fig, axs)

@torch.no_grad()
def sample_residual_probunet_model(model, dataloader, epoch, device, batch=None):
    """
    Generates and plots samples from the Probabilistic U-Net model, but shows residual predictions 
    instead of reconstructing them to high-resolution outputs.

    Args:
        model (ProbabilisticUNet): The trained Probabilistic U-Net model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        epoch (int): Current epoch number (for labeling plots).
        device (torch.device): Computation device (CPU or GPU).
        batch: If batch is provided, it uses the batch directly. If not, it takes a random batch from the dataloader.

    Returns:
        torch.Tensor: The generated residual predictions.
    """
    model.eval()
    if batch is None:
        batch = next(iter(dataloader))
    
    inputs = batch['inputs'][:2].to(device)  # Select 2 random low-resolution inputs
    lrinterp = batch['lrinterp'][:2].to(device)
    hr = batch['hr'][:2].to(device)
    timestamps = batch['timestamps'][:2].unsqueeze(dim=1).to(device)
    timestamps_float = batch['timestamps_float'][:2]

    num_samples = 3  # Number of predicted outputs per input
    residual_preds = []

    for _ in range(num_samples):
        # The model outputs residual predictions directly
        output = model(inputs, t=timestamps, training=False)
        residual_preds.append(output.cpu())

    # Stack the predictions: shape [batch_size, num_samples, nvars, H, W]
    residual_preds = torch.stack(residual_preds, dim=1)

    fig, axs = dataloader.dataset.plot_residual_sample_batch(
        lrinterp=lrinterp.cpu(),
        residual_preds=residual_preds.cpu(),
        hr=hr.cpu(),
        timestamps_float=timestamps_float,
        epoch=epoch,
        N=2,
        num_samples=num_samples
    )

    return residual_preds, (fig, axs)

@torch.no_grad()
def analyze_residual_contribution(model, dataloader, device, num_samples=10):
    """
    Analyze how much the predicted residuals actually improve over 
    simple bilinear interpolation.
    """
    model.eval()
    
    residual_mags = []
    lrinterp_errors = []  # Error of just using lrinterp
    model_errors = []     # Error of using lrinterp + predicted residual
    
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
            
        inputs = batch['inputs'].to(device)
        targets = batch['targets'].to(device)  # This is the TRUE residual
        hr = batch['hr'].to(device)
        lrinterp = batch['lrinterp'].to(device)
        timestamps = batch['timestamps'].unsqueeze(1).to(device)
        
        # Model prediction
        pred_residual = model(inputs, t=timestamps, training=False)
        
        # Reconstruct HR
        hr_pred = dataloader.dataset.residual_to_hr(
            pred_residual.cpu(), 
            lrinterp.cpu()
        ).to(device)
        
        # Metrics
        residual_mags.append(pred_residual.abs().mean().item())
        lrinterp_errors.append((hr - lrinterp).abs().mean().item())
        model_errors.append((hr - hr_pred).abs().mean().item())
    
    print(f"\n[ANALYSIS] Residual Contribution:")
    print(f"  Mean |predicted_residual|: {np.mean(residual_mags):.4f}")
    print(f"  Mean |true_residual|: {np.mean(lrinterp_errors):.4f}")
    print(f"  Error (lrinterp only): {np.mean(lrinterp_errors):.4f}")
    print(f"  Error (lrinterp + model): {np.mean(model_errors):.4f}")
    print(f"  Improvement: {np.mean(lrinterp_errors) - np.mean(model_errors):.4f}")
    print(f"  Improvement %: {100*(1 - np.mean(model_errors)/np.mean(lrinterp_errors)):.2f}%")
