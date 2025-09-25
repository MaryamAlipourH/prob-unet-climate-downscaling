import torch
import torch.nn as nn
from torch.distributions import Normal, Independent, kl
from prob_unet_utils import init_weights
from networks import UNet
import numpy as np
from prob_unet_utils import afcrps_loss, crps_loss
from prob_unet_utils import wmse_ms_ssim_loss   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AxisAlignedConvGaussian(nn.Module):
    """
    Axis-Aligned Convolutional Gaussian distribution for the latent space.
    This module computes the mean (mu) and log of standard deviation (log_sigma)
    of the Gaussian distribution using convolutional layers.
    """

    def __init__(self, input_channels, num_filters, latent_dim, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.posterior = posterior

        # If posterior, the input will include the target concatenated
        if self.posterior:
            self.input_channels += input_channels  

        
        self.contracting_path = nn.ModuleList()
        layers = []
        
        for i in range(len(self.num_filters)):
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = self.num_filters[i]

            if i != 0:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) 

            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(2): 
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))

        self.encoder = nn.Sequential(*layers)

        # Convolutional layers to compute mu and log_sigma
        self.conv_mu = nn.Conv2d(num_filters[-1], latent_dim, kernel_size=1)
        self.conv_log_sigma = nn.Conv2d(num_filters[-1], latent_dim, kernel_size=1)

        self.apply(init_weights)

    def forward(self, x, target=None):
        """
        Forward pass to compute the distribution of the latent variable.
        Args:
            x (torch.Tensor): Input tensor.
            target (torch.Tensor, optional): Target tensor (for posterior).
        Returns:
            dist (torch.distributions.Distribution): The computed Gaussian distribution.
        """
        # Concatenate input and target for posterior
        if self.posterior and target is not None:
            x = torch.cat([x, target], dim=1)

        # Encode the input to get the latent features
        h = self.encoder(x)

        # Global average pooling to get a single vector per sample
        h = torch.mean(h, dim=[2, 3], keepdim=True)

        # Compute mu and log_sigma
        mu = self.conv_mu(h)
        log_sigma = self.conv_log_sigma(h)

        # Remove the extra dimensions
        mu = mu.squeeze(-1).squeeze(-1)
        log_sigma = log_sigma.squeeze(-1).squeeze(-1)

        # Create a Normal distribution with the computed parameters
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma) + 1e-7), 1)
        return dist

class Fcomb(nn.Module):
    """
    Combines the UNet features with the latent variable z to produce the final output.
    """

    def __init__(self, unet_output_channels, latent_dim, num_classes):
        super(Fcomb, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2, 3]

        self.layers = nn.Sequential(
            nn.Conv2d(unet_output_channels + latent_dim, unet_output_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(unet_output_channels, unet_output_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(unet_output_channels, num_classes, kernel_size=1)
        )

        self.apply(init_weights)

    def tile(self, a, dim, n_tile):
        """
        This function mimics TensorFlow's `tile()` function for PyTorch.
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*repeat_idx)
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Forward pass to combine UNet features with latent variable.
        Args:
            feature_map (torch.Tensor): Feature map from UNet.
            z (torch.Tensor): Sampled latent variable.
        Returns:
            output (torch.Tensor): The final output tensor.
        """
        # Tile z to match feature map size
        z = torch.unsqueeze(z, 2)
        z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
        z = torch.unsqueeze(z, 3)
        z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

        # Concatenate feature map and latent variable
        feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
        output = self.layers(feature_map)
        return output

class ProbabilisticUNet(nn.Module):

    """
    The Probabilistic U-Net model combining a U-Net backbone with a variational latent space.
    """

    def __init__(self, input_channels, num_classes, latent_dim, num_filters, model_channels, channel_mult, beta_0, beta_1, beta_2):
        super(ProbabilisticUNet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        # Initialize the U-Net backbone
        self.unet = UNet(
            img_resolution=(128, 128),  
            in_channels=input_channels,
            out_channels=num_filters[0],
            label_dim=1,
            model_channels=model_channels,
            channel_mult=channel_mult,
            use_diffuse=False
        ).to(device)

        # Prior network (without target)
        self.prior = AxisAlignedConvGaussian(
            input_channels=input_channels,
            num_filters=num_filters,
            latent_dim=latent_dim,
            posterior=False
        ).to(device)

        # Posterior network (with target)
        self.posterior = AxisAlignedConvGaussian(
            input_channels=input_channels,
            num_filters=num_filters,
            latent_dim=latent_dim,
            posterior=True
        ).to(device)

        # Combines UNet features and latent variable to produce the output
        self.fcomb = Fcomb(
            unet_output_channels=num_filters[0],
            latent_dim=latent_dim,
            num_classes=num_classes
        ).to(device)

        # Apply Kaiming initialization to all the convolutional layers
        # self.apply(init_weights)  # It has already been applied to the individual components

    def forward(self, x, target=None, t=None, training=True):

        """
        Forward pass of the Probabilistic U-Net.

        Args:
            x (torch.Tensor): Input tensor.
            target (torch.Tensor, optional): Target tensor (for training).
            training (bool): Flag indicating whether in training mode.

        Returns:
            output (torch.Tensor): The model's output tensor.
        """

        # Get features from the UNet backbone      
        unet_features = self.unet(x)


        # During training, sample z from the posterior
        if training and target is not None:
            self.posterior_latent_space = self.posterior(x, target)
            z = self.posterior_latent_space.rsample()
    

        # During inference, sample z from the prior
        else:
            self.prior_latent_space = self.prior(x)
            z = self.prior_latent_space.rsample()
        
        output = self.fcomb(unet_features, z)
        return output
    
    # -----------------------------------------------------------------
    # ELBO with WMSE-MS-SSIM reconstruction term
    # -----------------------------------------------------------------
    def elbo(self, x, target, t,
             M: int = 1,                       # one posterior sample is enough here
             alpha_w: float = 0.007,
             beta_w:  float = 0.048,
             lam_w:   float = 0.000):
        """
        ELBO = β₀·recon + β₁·KL(q‖p) + β₂·KL(q‖𝒩(0,I))

        recon ≡ WMSE-MS-SSIM (Hess & Boers 2022).
        """
        # ─ encode ────────────────────────────────────────────────────
        unet_features          = self.unet(x)
        self.prior_latent_space     = self.prior(x)
        self.posterior_latent_space = self.posterior(x, target)

        # ─ draw M posterior samples and average their recon loss ────
        recon_losses = []
        for _ in range(M):
            z = self.posterior_latent_space.rsample()
            pred = self.fcomb(unet_features, z)        # [B,C,H,W]
            loss, wmse, msssim = wmse_ms_ssim_loss(pred, target,
                                     alpha=alpha_w, beta=beta_w, lam=lam_w, return_components=True)
            recon_losses.append(loss)
        recon_loss = torch.stack(recon_losses).mean()

        # ─ KL terms ─────────────────────────────────────────────────
        kl_div  = kl.kl_divergence(self.posterior_latent_space,
                                   self.prior_latent_space)
        standard_gaussian = Independent(
            Normal(loc=torch.zeros_like(self.posterior_latent_space.base_dist.loc),
                   scale=torch.ones_like(self.posterior_latent_space.base_dist.scale)), 1)
        

        # ─ total ────────────────────────────────────────────────────
        total = (self.beta_0 * recon_loss
               + self.beta_1 * kl_div.mean())

        # for logging keep the scalar recon-only value
        return total, [recon_loss.detach().cpu().item()], kl_div, wmse.detach().cpu().item(), msssim.detach().cpu().item()


    # # -----------------------------------------------------------------
    # # The ELBO using afCRPS loss
    # # -----------------------------------------------------------------
    # def elbo(self, x, target, t, M=5, alpha=0.95):
    #     """
    #     Compute the 'ELBO' with an afCRPS reconstruction term.
    #     """

    #     if M < 2:
    #         raise ValueError(f"M must be at least 2 to compute afCRPS but got M={M}")
    #     B = x.shape[0]

    #     # 1) Encode features / distributions
    #     unet_features = self.unet(x)
    #     self.prior_latent_space = self.prior(x)
    #     self.posterior_latent_space = self.posterior(x, target)

    #     # 2) Draw M samples from the posterior
    #     ensemble = []
    #     for _ in range(M):
    #         z_post = self.posterior_latent_space.rsample()
    #         pred_sample = self.fcomb(unet_features, z_post)  # shape [B, C, H, W]
    #         ensemble.append(pred_sample)
    #     # Stack to [B, M, C, H, W]
    #     ensemble_pred = torch.stack(ensemble, dim=1)

    #     # # 3) Compute afCRPS
    #     crps = afcrps_loss(ensemble_pred, target, alpha=alpha)
    #     # crps = crps_loss(ensemble_pred, target)

    #     # 4) KL divergences
    #     kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
    #     standard_gaussian = Independent(
    #         Normal(
    #             loc=torch.zeros_like(self.posterior_latent_space.base_dist.loc),
    #             scale=torch.ones_like(self.posterior_latent_space.base_dist.scale)
    #         ),
    #         1
    #     )
        

    #     # 5) Combine
    #     total_loss = (
    #         self.beta_0 * crps
    #       + self.beta_1 * kl_div.mean())

    #     # For logging, we might return them as well
    #     return total_loss, [crps.item()], kl_div
    

    
    # # -----------------------------------------------------------------
    # # The ELBO with L1 reconstruction term (original)
    # # -----------------------------------------------------------------
  
    # def elbo(self, x, target, t):

    #     """
    #     Computes the Evidence Lower Bound (ELBO) loss for training.

    #     Args:
    #         x (torch.Tensor): Input tensor.
    #         target (torch.Tensor): Target tensor.

    #     Returns:
    #         total_loss (torch.Tensor): The total ELBO loss.
    #         recon_loss (torch.Tensor): The reconstruction loss component.
    #         kl_div (torch.Tensor): The KL divergence component.
    #     """

    #      # Get features from the UNet backbone      
    #     unet_features = self.unet(x)

    #     # Compute prior and posterior distributions
    #     self.prior_latent_space = self.prior(x)
    #     self.posterior_latent_space = self.posterior(x, target)

    #     # Sample z from the posterior
    #     z_posterior = self.posterior_latent_space.rsample()

    #     # Compute the output
    #     output = self.fcomb(unet_features, z_posterior)

    #     # Initialize total reconstruction loss and list for individual variable losses
    #     total_recon_loss = 0
    #     recon_loss_list = []

    #     for i in range(output.shape[1]):  
    #         # Compute reconstruction loss for each variable
    #         recon_loss = nn.L1Loss(reduction='mean')(output[:, i, :, :], target[:, i, :, :])
    #         recon_loss_list.append(recon_loss.item())  # Store individual variable loss
        
    #     total_recon_loss = nn.L1Loss()(output, target)  # Average to total loss

    #     # KL divergence between posterior and prior
    #     kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)

    #     # Define the standard Gaussian distribution
    #     standard_gaussian = Independent(
    #         Normal(
    #             loc=torch.zeros_like(self.posterior_latent_space.base_dist.loc).to(device),
    #             scale=torch.ones_like(self.posterior_latent_space.base_dist.scale).to(device)
    #         ),
    #         1
    #     )

    #     # KL divergence between posterior and standard Gaussian
    #     kl_div2 = kl.kl_divergence(self.posterior_latent_space, standard_gaussian)

    #     total_loss = self.beta_0 * total_recon_loss + self.beta_1 * torch.mean(kl_div) + self.beta_2 * torch.mean(kl_div2)

    #     return total_loss, recon_loss_list, kl_div, kl_div2


    
