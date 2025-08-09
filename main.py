import torch
import matplotlib.pyplot as plt
# from dask.distributed import Client

import climex_utils as cu
import train_prob_unet_model as tm  
from prob_unet import ProbabilisticUNet
from prob_unet_utils import plot_losses, plot_losses_mae
import pickle
import numpy as np
import random
import os
  

if __name__ == "__main__":

    def set_seed(seed):
        random.seed(seed) 
        np.random.seed(seed)  
        torch.manual_seed(seed) 
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False  
        os.environ['PYTHONHASHSEED'] = str(seed)

    # -- 1) Set seed for reproducibility
    set_seed(42)  

    # -- 2) Importing all required arguments
    args = tm.get_args()
    args.lowres_scale = 16
    args.batch_size = 32
    args.num_epochs = 10

    # Initialize the Probabilistic UNet model
    probunet_model = ProbabilisticUNet(
        input_channels=len(args.variables),
        num_classes=len(args.variables),
        latent_dim=16,
        num_filters=[32, 64, 128, 256],
        model_channels=32,
        channel_mult=[1, 2, 4, 8],
        beta_0=0.0,
        beta_1=0.0,
        beta_2=0.0  
    ).to(args.device)

    # -- 3) Prepare datasets
    dataset_train = cu.climex2torch(
        datadir=args.datadir,
        years=args.years_train,
        variables=args.variables,
        coords=args.coords,
        lowres_scale=args.lowres_scale,
        type="lrinterp_to_residuals",
        transfo=True
    )
    dataset_val = cu.climex2torch(
        datadir=args.datadir,
        years=args.years_val,
        variables=args.variables,
        coords=args.coords,
        lowres_scale=args.lowres_scale,
        type="lrinterp_to_residuals",
        transfo=True
    )
    dataset_test = cu.climex2torch(
        datadir=args.datadir,
        years=args.years_test,
        variables=args.variables,
        coords=args.coords,
        lowres_scale=args.lowres_scale,
        type="lrinterp_to_residuals",
        transfo=True
    )

    # -- 4) Build DataLoaders
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    dataloader_test_random = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    # -- 5) Define optimizer
    optimizer = args.optimizer(params=probunet_model.parameters(), lr=args.lr)
    # Example alternative:
    # optimizer = torch.optim.Adam(probunet_model.parameters(), lr=args.lr, weight_decay=1e-4)

    # -- 6) We track CRPS, KL, KL2 each epoch for train/val
    train_crps_list, train_kl_list, train_kl2_list = [], [], []
    val_crps_list,   val_kl_list,   val_kl2_list   = [], [], []

    # For convenience, we keep your adaptive betas:
    beta_0 = 1.0
    beta_1 = 0.0
    beta_2 = 0.0
    warmup_epochs = 2

    print(f"Probabilistic Unet Latent dim: {probunet_model.latent_dim}")

    # -- 7) Main training loop
    for epoch in range(1, args.num_epochs + 1):
        # Set model betas
        probunet_model.beta_0 = beta_0
        probunet_model.beta_1 = beta_1
        probunet_model.beta_2 = beta_2

        print(f"Epoch {epoch}/{args.num_epochs} - beta_0: {probunet_model.beta_0:.4f}, "
              f"beta_1: {probunet_model.beta_1:.4f}, beta_2: {probunet_model.beta_2:.4f}")

        # 7a) Train for one epoch (returns mean_crps, mean_kl, mean_kl2)
        train_crps, train_kl, train_kl2 = tm.train_probunet_step(
            model=probunet_model,
            dataloader=dataloader_train,
            optimizer=optimizer,
            epoch=epoch,
            num_epochs=args.num_epochs,
            device=args.device,       
            ensemble_size=1    # how many samples per forward pass
        )
        train_crps_list.append(train_crps)
        train_kl_list.append(train_kl)
        train_kl2_list.append(train_kl2)

        # 7b) Update betas after warmup
        if epoch > warmup_epochs:
            # beta_0 = 1.0 / (train_crps + 1e-7)
            beta_0 = 1.0
            beta_1 = 1.0 / (train_kl   + 1e-7)
            # beta_2 = 1.0 / (train_kl2  + 1e-7)
            beta_2 = 0.0
        else:
            beta_0, beta_1, beta_2 = 1.0, 0.0, 0.0

        # 7c) Evaluate on validation data
        val_crps, val_kl, val_kl2 = tm.eval_probunet_model(
            model=probunet_model,
            dataloader=dataloader_val,
            device=args.device,
            ensemble_size=1
        )
        val_crps_list.append(val_crps)
        val_kl_list.append(val_kl)
        val_kl2_list.append(val_kl2)

        print(f"[Train] CRPS={train_crps:.4f}, KL={train_kl:.4f}, KL2={train_kl2:.4f} | "
              f"[Val] CRPS={val_crps:.4f}, KL={val_kl:.4f}, KL2={val_kl2:.4f}")

        # 7d) Example sampling from the model for sanity checks
        test_batch = next(iter(dataloader_test_random))

        # Residual predictions
        residual_preds, (fig, axs) = tm.sample_residual_probunet_model(
            model=probunet_model,
            dataloader=dataloader_test_random,
            epoch=epoch,
            device=args.device,
            batch=test_batch
        )
        fig.savefig(f"{args.plotdir}/epoch{epoch}_residuals.png", dpi=300)
        plt.close(fig)

        fig_difs, axs_difs = dataset_test.plot_residual_differences(
            residual_preds=residual_preds,
            timestamps_float=test_batch['timestamps_float'][:2],
            epoch=epoch,
            N=2, 
            num_samples=3
        )
        fig_difs.savefig(f"{args.plotdir}/epoch{epoch}_res_difs.png", dpi=300)
        plt.close(fig_difs)

        # Full reconstruction
        samples, (fig, axs) = tm.sample_probunet_model(
            model=probunet_model,
            dataloader=dataloader_test_random,
            epoch=epoch,
            device=args.device,
            batch=test_batch
        )
        fig.savefig(f"{args.plotdir}/epoch{epoch}_reconstructed.png", dpi=300)
        plt.close(fig)

    # -- 8) Save final model weights
    torch.save(probunet_model.state_dict(),
               f"{args.plotdir}/probunet_model_lat_dim_{probunet_model.latent_dim}.pth")

    # -- 9) Save losses for analysis
    losses_to_save = {
        "train_crps": train_crps_list,
        "train_kl":   train_kl_list,
        "train_kl2":  train_kl2_list,
        "val_crps":   val_crps_list,
        "val_kl":     val_kl_list,
        "val_kl2":    val_kl2_list,
    }
    with open(f"{args.plotdir}/losses.pkl", "wb") as f:
        pickle.dump(losses_to_save, f)

  
    epochs = np.arange(1, args.num_epochs+1)
    plt.plot(epochs, train_crps_list, label='Train CRPS')
    plt.plot(epochs, val_crps_list,   label='Val CRPS', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('CRPS')
    plt.legend()
    plt.title('Training and Validation CRPS')
    plt.savefig(f"{args.plotdir}/CRPS_curve.png", dpi=300)
    plt.close()
