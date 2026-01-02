import numpy as np

import torch
import torch.nn as nn

import climex_utils as cu
import trainmodel as tm
import baseline.deterministic_unet as det_unet

def BCSD(datatrain, datatest, epsilon=1e-9):

    train_nyears, test_nyears = len(datatrain.years), len(datatest.years)
    scaling_years = min(train_nyears, test_nyears)
    #datatrain_xr = cu.climexEDA(datatrain_tensor.datadir, years=datatrain_tensor.years[-scaling_years:], coords=datatrain_tensor.coords)

    sc_num = datatrain.data[:, -scaling_years*365:, :, :].groupby('time.dayofyear').mean()
    sc_num = torch.from_numpy(sc_num.data).repeat(1, scaling_years, 1, 1).transpose(0,1)

    data_denom = datatrain.lrinterp[-scaling_years*365:, :, :, :]
    dayofyear_idx = []
    for d in range(365):
        dayofyear_idx.append(np.arange(scaling_years) * 365 + d)
    sc_denom = torch.cat([data_denom[idx] for idx in dayofyear_idx])

    bcsd = datatest.lrinterp * sc_num / (sc_denom + epsilon)
    return bcsd

class LinearCNN(nn.Module):

    def __init__(self, resolution, in_channels, ds_factor, device=None):
        super(LinearCNN, self).__init__()

        self.device = device
        
        first_kernel = 3
        first_padding = int((first_kernel - 1) / 2)
        latent_channels = 10

        self.first_conv = nn.Conv2d(in_channels=in_channels, out_channels=latent_channels, kernel_size=first_kernel, stride=1, padding=first_padding)
        self.second_conv = nn.Conv2d(in_channels=latent_channels, out_channels=in_channels, kernel_size=first_kernel, stride=1, padding=first_padding)

        self.timefirstlinear = nn.Linear(in_features=1, out_features=128, bias=True)
        self.timesecondlinear = nn.Linear(in_features=128, out_features=512, bias=True)
        self.timethirdlinear = nn.Linear(in_features=512, out_features=4096, bias=True)

    def forward(self, x, class_labels):

        """
        bs = x.size(0)

        t = class_labels.to(torch.float32)
        t = self.timefirstlinear(t)
        t = self.timesecondlinear(t)
        t = self.timethirdlinear(t)
        t = t.reshape(bs, 1, 64, 64)
        """

        #x = torch.cat([x, t], dim=1)
        x = self.first_conv(x)
        x = self.second_conv(x)

        return x
