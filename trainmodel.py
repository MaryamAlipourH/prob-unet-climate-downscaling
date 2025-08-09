import os
import warnings
import argparse
import torch
import numpy as np

import climex_utils as cu 

from tqdm import tqdm
from datetime import datetime

warnings.filterwarnings('ignore')

def get_args():
    """
    This function returns a dictionary containing all necessary arguments for importing ClimEx data, training, evaluating, and sampling from a downscaling ML model.
    This function is helpful for doing sweeps and performing hyperparameter tuning.
    """
    parser = argparse.ArgumentParser()

    # climate dataset arguments
    parser.add_argument('--datadir', type=str, default='/home/julie/Data/Climex/day/kdj/')
    parser.add_argument('--variables', type=list, default=['pr', 'tasmin', 'tasmax'])
    parser.add_argument('--years_train', type=range, default=range(1960, 2020))
    parser.add_argument('--years_val', type=range, default=range(2020, 2040))
    parser.add_argument('--years_test', type=range, default=range(2040, 2060))
    parser.add_argument('--coords', type=list, default=[120, 184, 120, 184])
    parser.add_argument('--resolution', type=tuple, default=(64, 64))
    parser.add_argument('--lowres_scale', type=int, default=8)
    parser.add_argument('--timetransform', type=str, default='id', choices=['id', 'cyclic'])

    # Downscaling method 
    parser.add_argument('--ds_model', type=str, default='deterministic_unet', choices=['deterministic_unet', 'probabilistic_unet', 'vae', 'linearcnn', 'bcsd'])

    # ML training arguments
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--accum', type=int, default=8)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--optimizer', type=object, default=torch.optim.AdamW)

    # WandB activation 
    parser.add_argument('--wandb', type=bool, default=False)

    # GPU
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Save results arguments
    strtime = datetime.now().strftime('%m/%d/%Y%H')
    plotdir = './results/plots/' + strtime + '/'
    parser.add_argument('--plotdir', type=str, default=plotdir)
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)
    checkpoints_dir = './results/checkpoints/' + strtime +  '/'
    parser.add_argument('--checkpoints_dir', type=str, default=checkpoints_dir)
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    args, _ = parser.parse_known_args()

    return args


# Probabilistic generalization of MAE
def crps_empirical(pred, truth):
    """
    Code borrowed from pyro: https://docs.pyro.ai/en/stable/_modules/pyro/ops/stats.html#crps_empirical

    Computes negative Continuous Ranked Probability Score CRPS* [1] between a
    set of samples ``pred`` and true data ``truth``. This uses an ``n log(n)``
    time algorithm to compute a quantity equal that would naively have
    complexity quadratic in the number of samples ``n``::

        CRPS* = E|pred - truth| - 1/2 E|pred - pred'|
              = (pred - truth).abs().mean(0)
              - (pred - pred.unsqueeze(1)).abs().mean([0, 1]) / 2

    Note that for a single sample this reduces to absolute error.

    **References**

    [1] Tilmann Gneiting, Adrian E. Raftery (2007)
        `Strictly Proper Scoring Rules, Prediction, and Estimation`
        https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

    :param torch.Tensor pred: A set of sample predictions batched on rightmost dim.
        This should have shape ``(num_samples,) + truth.shape``.
    :param torch.Tensor truth: A tensor of true observations.
    :return: A tensor of shape ``truth.shape``.
    :rtype: torch.Tensor
    """
    if pred.shape[1:] != (1,) * (pred.dim() - truth.dim() - 1) + truth.shape:
        raise ValueError(
            "Expected pred to have one extra sample dim on left. "
            "Actual shapes: {} versus {}".format(pred.shape, truth.shape)
        )
    opts = dict(device=pred.device, dtype=pred.dtype)
    num_samples = pred.size(0)
    if num_samples == 1:
        return (pred[0] - truth).abs()

    pred = pred.sort(dim=0).values
    diff = pred[1:] - pred[:-1]
    weight = torch.arange(1, num_samples, **opts) * torch.arange(
        num_samples - 1, 0, -1, **opts
    )
    weight = weight.reshape(weight.shape + (1,) * (diff.dim() - 1))

    return (pred - truth).abs().mean(0) - (diff * weight).sum(0) / num_samples**2

class CRPSLoss(torch.nn.Module):
    def __init__(self):
        super(CRPSLoss, self).__init__()

    def forward(self, pred, truth):
        return crps_empirical(pred, truth)

def train_step(model, dataloader, loss_fn, optimizer, scaler, epoch, num_epochs, accum, act_wandb, device):

    """
    Function for training the UNet model for a single epoch.

    model: instance of the Unet class
    dataloader: torch training dataloader
    loss_fn: loss function
    optimizer: torch optimizer 
    scaler: scaler for mixed precision training
    epoch: current epoch
    num_epochs: total number of epochs
    accum: number of steps to accumulate gradients over
    wandb: True if wandb activated
    device: device to use (GPU)

    return -> average loss value over the epoch
    """

    model.train()

    # Activating progress bar
    with tqdm(total=len(dataloader), dynamic_ncols=True) as tq:
        tq.set_description(f'Train :: Epoch: {epoch}/{num_epochs}')

        # running_losses = dict.fromkeys(["pr", "tasmin", "tasmax"], [])
        running_losses = {var: [] for var in ["pr", "tasmin", "tasmax"]}
        running_loss = []
        # Looping over the entire dataloader set
        for i, batch in enumerate(dataloader):
            tq.update(1)

            optimizer.zero_grad()

            # Extracting training data from batch
            inputs, targets = batch['inputs'].to(device), batch['targets'].to(device)
            timestamps = batch['timestamps'].unsqueeze(dim=1).to(device)

            # Performing forward pass and computing loss
            preds = model(inputs, class_labels=timestamps)
            loss = loss_fn(preds, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            """
            UNCOMMENT FOR MIXED PRECISION TRAINING

            # Performing forward pass of the unet model and computing loss
            with torch.cuda.amp.autocast():
                preds = model(inputs, class_labels=timestamps)
                loss = loss_fn(preds, targets)

            # Logging training loss in wandb
            if act_wandb:
                wandb.log(data={"train-loss": loss.item()})

            # Performing backprograpation
            scaler.scale(loss).backward()

            # Updating optimizer every accum steps
            if (i + 1) % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            """

            # Logging per-variable training losses
            loss_pr = loss_fn(preds[:,0,:,:], targets[:,0,:,:])
            loss_tasmin = loss_fn(preds[:,1,:,:], targets[:,1,:,:])
            loss_tasmax = loss_fn(preds[:,2,:,:], targets[:,2,:,:])

            running_losses['pr'] = running_losses['pr'] + [loss_pr.item()]
            running_losses['tasmin'] = running_losses['tasmin'] + [loss_tasmin.item()]
            running_losses['tasmax'] = running_losses['tasmax'] + [loss_tasmax.item()]

            tq.set_postfix_str(s=f'Loss: {(loss.item()):.4f}')
            running_loss.append(loss.item())

        mean_loss = sum(running_loss) / len(running_loss)
        tq.set_postfix_str(s=f'Loss: {mean_loss:.4f}')

        return running_losses

@torch.no_grad()
def sample_model(model, dataloader, epoch, device):

    """
    Function for sampling from the unet model and plotting results.

    model: instance of the Unet class
    dataloader: torch dataloader (should be shuffled)
    epoch: last training epoch
    device: device to use (GPU)

    return -> predicted high-resolution samples, plots
    """

    model.eval()

    # Extracting data from the first batch of the dataloader
    batch = next(iter(dataloader))
    inputs, lrinterp, hr, timestamps, stand_stats = (batch['inputs'].to(device), batch['lrinterp'], batch['hr'], batch['timestamps'], batch['stand_stats'])

    # Performing forward pass on the unet
    residual_preds = model(inputs, class_labels=timestamps.unsqueeze(dim=1).to(device))

    # Converting predicted residual to high-resolution sample
    hr_pred = dataloader.dataset.residual_to_hr(residual_preds.detach().cpu(), lrinterp, stand_stats)

    # Plotting the results
    fig, axs = dataloader.dataset.plot_batch(lrinterp, hr_pred, hr, timestamps, epoch)

    return hr_pred, (fig, axs)

@torch.no_grad()
def eval_model(model, dataloader, loss_fn, reconstruct, wandb, device):

    """
    Function for evaluating the unet model.

    model: instance of the Unet class
    dataloader: torch dataloader 
    metric: metric used for evaluation
    wandb: True if wandb activated
    device: device to use (GPU)

    return -> averaged loss over the dataloader set
    """

    model.eval()

    # Activating progress bar
    with tqdm(total=len(dataloader), dynamic_ncols=True) as tq:
        tq.set_description(':: Evaluation ::')

        running_losses = dict.fromkeys(["pr", "tasmin", "tasmax"], [])
        # Looping over the entire dataloader set
        for i, batch in enumerate(dataloader):
            tq.update(1)

            # Extracting training data from batch and performing forward pass
            inputs, targets, timestamps, hr = (batch['inputs'].to(device), batch['targets'].to(device), batch['timestamps'], batch['hr'])
            residual_preds = model(inputs, class_labels=timestamps.unsqueeze(dim=1).to(device))

            # if we want to compute the loss on the high-resolution data
            if reconstruct:

                preds = dataloader.dataset.residual_to_hr(residual_preds.detach().cpu(), batch['lrinterp'], batch['stand_stats'])
                
                preds_hr = cu.kgm2sTommday(preds[:, 0, :, :])
                preds_tasmin = cu.KToC(preds[:, 1, :, :])
                preds_tasmax = cu.KToC(preds[:, 2, :, :])
                #preds_hr = cu.kgm2sTommday(cu.log_inv(preds[:, 0, :, :]))
                #preds_tasmin = cu.KToC(preds[:, 1, :, :])
                #preds_tasmax = cu.KToC(cu.log_inv(preds[:, 2, :, :]) + preds[:, 1, :, :])

                hr_pr = cu.kgm2sTommday(hr[:, 0, :, :])
                hr_tasmin = cu.KToC(hr[:, 1, :, :])
                hr_tasmax = cu.KToC(hr[:, 2, :, :])
                #hr_pr = cu.kgm2sTommday(cu.log_inv(hr[:, 0, :, :]))
                #hr_tasmin = cu.KToC(hr[:, 1, :, :])
                #hr_tasmax = cu.KToC(cu.log_inv(hr[:, 2, :, :]) + hr[:, 1, :, :])

                loss_pr = loss_fn(preds_hr, hr_pr)
                loss_tasmin = loss_fn(preds_tasmin, hr_tasmin)
                loss_tasmax = loss_fn(preds_tasmax, hr_tasmax)

                running_losses['pr'] = running_losses['pr'] + [loss_pr.item()]
                running_losses['tasmin'] = running_losses['tasmin'] + [loss_tasmin.item()]
                running_losses['tasmax'] = running_losses['tasmax'] + [loss_tasmax.item()]
            
            # if we want to compute the loss directly on the residual
            else:

                # Reconstructing predictions to high-resolution
                loss_pr = loss_fn(residual_preds[:,0,:,:], targets[:,0,:,:])
                loss_tasmin = loss_fn(residual_preds[:,1,:,:], targets[:,1,:,:])
                loss_tasmax = loss_fn(residual_preds[:,2,:,:], targets[:,2,:,:])

                running_losses['pr'] = running_losses['pr'] + [loss_pr.item()]
                running_losses['tasmin'] = running_losses['tasmin'] + [loss_tasmin.item()]
                running_losses['tasmax'] = running_losses['tasmax'] + [loss_tasmax.item()]

    return running_losses
