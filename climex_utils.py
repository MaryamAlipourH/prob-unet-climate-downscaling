import glob
import dask
from dask.distributed import Client
import xarray as xr
import numpy as np
import bottleneck
import cftime

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cartopy import crs as ccrs

import torch
import torch.nn as nn
from torch.utils.data import Dataset

#####

# Converts np.datetime64 to np.float64 (number of days to date)
def date_to_float(date_array):
    return date_array.values.astype(float)

# # Converts np.float64 to np.datetime64
# def float_to_date(float_array):
#     return np.array(float_array, dtype="datetime64[ns]")
def float_to_date(time_value):
    # Adjust based on the units of timestamps_float
    return np.datetime64(int(time_value), 'ns')  # If time_value is in nanoseconds

# For precipitation
def kgm2sTommday(data):
    return data*24*60*60

# For inverse transformation
def softplus_inv(data, threshold=20., c=1e-7):
    mask = data > threshold
    data[mask] = data[mask]
    data[~mask] = torch.log(torch.exp(data[~mask] + c) - 1.)
    return data

def softplus(data, threshold=20., c=1e-7):
    mask = data > threshold
    data[mask] = data[mask]
    data[~mask] = torch.log(torch.exp(data[~mask]) + 1.) - c
    return data
        
# For temperature
def KToC(data):
    return data - 273.15

####

class climex2torch(Dataset):

    """
    Dataset class that loads and converts data from NetCDF files to a Pytorch tensor on initialization. 
    climex2torch object can be fed to a Pytorch Dataloader.
    """

    def __init__(self, datadir, years=range(1960, 2020), variables=["pr", "tasmin", "tasmax"], coords=[120, 184, 120, 184], type="lr_to_hr", lowres_scale = 4, transfo=False, megafile=None):

        """
        datadir: (str) path to the directory containing NetCDF files;
        years: (list of int) indicates which years should climex2torch import data from;
        variables: (list of str) indicates what variables should climex2torch import data from;
        coords: (list of int) (form: [start_rlon, end_rlon, start_rlat, end_rlat]) climex2torch will only import data from the resulting window;
        type: (str) indicates the data pipeline to train the model on (lr_to_hr, lr_to_residuals, lrinterp_to_residuals, lrinterp_to_hr);
        lowres_scale: (int) downscaling factor;
        """

        super().__init__()

        # Setup dask distributed cluster
        client = Client()

        self.datadir = datadir
        self.years = years
        self.variables = variables
        self.nvars = len(variables)
        self.coords = coords
        self.type = type
        self.lowres_scale = lowres_scale
        self.transfo = transfo
        self.megafile = megafile
        self.epsilon = 1e-10 #used for standardization
        self.lrstats = None #used for standardization

        # Preprocessing function to select only desired coordinates
        def select_coords(ds):
            return ds.isel(rlon=slice(coords[0], coords[1]), rlat=slice(coords[2],coords[3]))
        
        if megafile is None:

            # Recursively getting all NetCDF files names
            files = []
            for year in self.years:
                for var in variables:
                    files.append(glob.glob("{path}/*_{var}_*_{year}_*".format(path=self.datadir, var=var, year=year))[0])  

            print("Opening and lazy loading netCDF files")  

            # Importing all NetCDF files into a xarray Dataset with lazy loading
            self.data = xr.open_mfdataset(paths=files, engine='h5netcdf', preprocess=select_coords, data_vars="minimal", coords="minimal", compat="override", parallel=True)[self.variables]

        else:

            print("Opening and lazy loading megafile")
            self.data = xr.open_dataset(self.megafile, engine="h5netcdf")[self.variables]
        
        # Extracting latitude and longitude data (for plotting function) and timestamps
        self.lon = self.data.lon
        self.lat = self.data.lat

        # Extracting time features
        time = self.data.indexes["time"].to_datetimeindex()
        month = np.sin(2*np.pi*time.month/12)
        day = np.cos(2*np.pi*time.day/31)
        self.timestamps = torch.from_numpy(np.array(month + day)).float()
        self.timestamps_float = date_to_float(time)

        data_temp = self.data

        # Dropping unnecessary variables and encoding
        data_temp = data_temp.drop_vars(["lat", "lon"]).drop_indexes(["rlon", "rlat"]).drop_encoding().to_array()

        print("Loading dataset into memory")
        data_temp.load()

        print("Converting xarray Dataset to Pytorch tensor")

        # Loading into memory high-resolution ground-truth data from desired spatial window and converting to Pytorch tensor (time, nvar, height, width)
        self.hr = torch.from_numpy(data_temp.to_numpy()).transpose(0, 1)

        # Tranformations (prep > 0 and tmax > tmin)
        if self.transfo:
            self.hr[:, 0, :, :] = softplus_inv(self.hr[:, 0, :, :])
            self.hr[:, 2, :, :] = softplus_inv(self.hr[:, 2, :, :] - self.hr[:, 1, :, :], c=0.)

        client.close()

        print("")
        print("##########################################")
        print("############ PROCESSING DONE #############")
        print("##########################################")
        print("")


    def __len__(self):
         return len(self.timestamps)

    def __getitem__(self, idx):

        if self.type == "lr_to_hr":

            hr = self.hr[idx]
            lr = nn.AvgPool2d(kernel_size=self.lowres_scale)(self.hr[idx])

            # If standardization statistics are not computed yet, compute them
            if self.lrstats is None :
                print("Computing statistics for standardization")
                self.lrstats = self.compute_stats()

            lr_stand = (lr - self.lrstats[0][0]) / (self.lrstats[0][1] + self.epsilon)
            hr_stand = (hr - self.lrstats[1][0]) / (self.lrstats[1][1] + self.epsilon)

            return {"inputs": lr_stand,
                    "targets": hr_stand,
                    "timestamps": self.timestamps[idx],
                    "timestamps_float": self.timestamps_float[idx],
                    "hr": hr, 
                    "lr": lr}
        
        if self.type == "lr_to_residuals":

            hr = self.hr[idx]
            lr = nn.AvgPool2d(kernel_size=self.lowres_scale)(self.hr[idx])

            # If standardization statistics are not computed yet, compute them
            if self.lrstats is None :
                print("Computing statistics for standardization")
                self.lrstats = self.compute_stats()

            lr_stand = (lr - self.lrstats[0][0]) / (self.lrstats[0][1] + self.epsilon)
            hr_stand = (hr - self.lrstats[1][0]) / (self.lrstats[1][1] + self.epsilon)

            residual = hr_stand - nn.functional.interpolate(input=lr_stand.unsqueeze(0), scale_factor=self.lowres_scale).squeeze()

            return {"inputs": lr_stand,
                    "targets": residual,
                    "timestamps": self.timestamps[idx],
                    "timestamps_float": self.timestamps_float[idx],
                    "hr": hr, 
                    "lr": lr,
                    "lrinterp": nn.functional.interpolate(input=lr.unsqueeze(0), scale_factor=self.lowres_scale).squeeze()}
        
        elif self.type == "lrinterp_to_residuals":

            hr = self.hr[idx].clone()

            # Low-resolution data is obtained by averaging the high-resolution data and then upsampling it
            lr = nn.AvgPool2d(kernel_size=self.lowres_scale)(hr)
            # lr[0, :, :] = torch.log(lr[0, :, :] + torch.tensor(1e-5)) - torch.log(torch.tensor(1e-5))
            lrinterp = nn.functional.interpolate(input=lr.unsqueeze(0), scale_factor=self.lowres_scale).squeeze() 

            # hr[0] = torch.log(lr[0, :, :] + torch.tensor(1e-5)) - torch.log(torch.tensor(1e-5))

            # If standardization statistics are not computed yet, compute them
            if self.lrstats is None :
                print("Computing statistics for standardization")
                self.lrstats = self.compute_stats()

            lrinterp_stand = (lrinterp - self.lrstats[1][0]) / (self.lrstats[1][1] + self.epsilon)
            hr_stand = (hr - self.lrstats[1][0]) / (self.lrstats[1][1] + self.epsilon)

            residual = hr_stand - lrinterp_stand
            timestamp = self.timestamps[idx]

            return {"inputs": lrinterp_stand,
                    "targets": residual,
                    "timestamps": timestamp,
                    "timestamps_float": self.timestamps_float[idx],
                    "hr": hr, 
                    "lr": lr,
                    "lrinterp": lrinterp}

        elif self.type == "lrinterp_to_hr":

            hr = self.hr[idx]

            # Low-resolution data is obtained by averaging the high-resolution data and then upsampling it
            lr = nn.AvgPool2d(kernel_size=self.lowres_scale)(hr)
            lrinterp = nn.functional.interpolate(input=lr.unsqueeze(0), scale_factor=self.lowres_scale).squeeze() 

            # If standardization statistics are not computed yet, compute them
            if self.lrstats is None :
                print("Computing statistics for standardization")
                self.lrstats = self.compute_stats()

            lrinterp_stand = (lrinterp - self.lrstats[1][0]) / (self.lrstats[1][1] + self.epsilon)
            hr_stand = (hr - self.lrstats[1][0]) / (self.lrstats[1][1] + self.epsilon) 

            timestamp = self.timestamps[idx]

            return {"inputs": lrinterp_stand,
                    "targets": hr_stand,
                    "timestamps": timestamp,
                    "timestamps_float": self.timestamps_float[idx],
                    "hr": hr, 
                    "lr": lr,
                    "lrinterp": lrinterp}


    # Computes the statistics of the low-resolution data for standardization
    def compute_stats(self):

        lr = nn.AvgPool2d(kernel_size=self.lowres_scale)(self.hr)

        mean, std = lr.mean(dim=0), lr.std(dim=0) 
        # Extend the dimension to match high-resolution
        mean_hrdim = mean.repeat_interleave(repeats=self.lowres_scale, dim=1).repeat_interleave(repeats=self.lowres_scale, dim=2)
        std_hrdim = std.repeat_interleave(repeats=self.lowres_scale, dim=1).repeat_interleave(repeats=self.lowres_scale, dim=2)

        return (mean, std), (mean_hrdim, std_hrdim)

    # Computes the inverse of the standardization for the residual
    def invstand_residual(self, standardized_residual):
        if self.type == "lr_to_hr" or self.type == "lrinterp_to_hr":
            return standardized_residual * (self.lrstats[1][1] + self.epsilon) + self.lrstats[1][0]
        elif self.type == "lrinterp_to_residuals" or self.type == "lr_to_residuals":
            return standardized_residual * (self.lrstats[1][1] + self.epsilon)
    
    # Adds the predicted residual to the input upsampled high-resolution
    def residual_to_hr(self, residual, lrinterp):
        return lrinterp + self.invstand_residual(residual)
   
    # Plot a batch (N) of samples (upsampled low-resolution, predicted high-resolution, groundtruth high-resolution)
    def plot_batch(self, lrinterp, hr_pred, hr, timestamps, epoch, N=2):

        # Initializing Plate Carrée and Rotated Pole projections (for other projections see https://scitools.org.uk/cartopy/docs/latest/reference/crs.html)
        rotatedpole_prj = ccrs.RotatedPole(pole_longitude=83.0, pole_latitude=42.5)
        platecarree_proj = ccrs.PlateCarree()

        # Initializing figure and subfigures (one subfigure per date)
        fig = plt.figure(figsize=(N * 18, 12), constrained_layout=True)
        subfigs = fig.subfigures(1, N, wspace=0.05)

        # Different colormaps for different type of climate variables
        prep_colors = [
            (1., 1., 1.), 
            (0.5, 0.88, 1.),
            (0.1, 0.15, 0.8),
            (0.39, 0.09, 0.66), 
            (0.85, 0.36, 0.14),
            (0.99, 0.91, 0.3)
        ]
        prep_colormap = mpl.colors.LinearSegmentedColormap.from_list(name="prep", colors=prep_colors)
        cmaps = {'pr': prep_colormap, 'temp': cm.get_cmap('RdBu_r'), 'error': cm.get_cmap('gist_heat_r')}

        axs = []
        # Batch (N) plotting loop      
        for j in range(N):

            axs.append(subfigs[j].subplots(self.nvars, 4, subplot_kw={'projection': rotatedpole_prj}, gridspec_kw={'wspace':0.01, 'hspace':0.005}))

            # Extracting latitude and longitude data corresponding to the j-th sample from the batch
            lat, lon = self.lat.sel(time=str(float_to_date(timestamps[j]))[:10]).load().to_numpy().squeeze(), self.lon.sel(time=str(float_to_date(timestamps[j]))[:10]).load().to_numpy().squeeze()

            # Variables plotting loop
            temp_max_abs = []
            temp_ims = []
            for i in range(self.nvars):

                if self.variables[i] == "pr":

                    cmap = cmaps["pr"]
                    unit = " (mm/day)"

                    # Converting units in mm/day and computing scaling values for colormap
                    if self.transfo:
                        lr_sample = kgm2sTommday(softplus(lrinterp[j,i]))
                        hr_pred_sample = kgm2sTommday(softplus(hr_pred[j,i]))
                        hr_sample = kgm2sTommday(softplus(hr[j,i]))
                    else:
                        lr_sample = kgm2sTommday(lrinterp[j,i])
                        hr_pred_sample = kgm2sTommday(hr_pred[j,i])
                        hr_sample = kgm2sTommday(hr[j,i])
                    vmin, vmax = 0, max(torch.amax(lr_sample), torch.amax(hr_pred_sample), torch.amax(hr_sample))

                    # Computing absolute error and setting corresponding vmin, vmax
                    error_sample = torch.abs(hr_sample - hr_pred_sample) 
                    err_vmin, err_vmax = 0, torch.amax(error_sample)

                    # Setting cartopy features on the Axes objects
                    for l in range(4):
                        axs[j][i, l].coastlines()
                        gl = axs[j][i, l].gridlines(crs=platecarree_proj, draw_labels=True, x_inline=False, y_inline=False, linestyle="--")
                        gl.top_labels = False
                        gl.right_labels = False
                        if l > 0: 
                            gl.left_labels = False

                    # Plotting samples in the following order: upsampled low-resolution, predicted high-resolution, groundtruth high-resolution
                    axs[j][i, 0].pcolormesh(lon, lat, lr_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)
                    axs[j][i, 1].pcolormesh(lon, lat, hr_pred_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)
                    im = axs[j][i, 2].pcolormesh(lon, lat, hr_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)

                    # Plotting the colorbar for the row with the correct label
                    cbar = plt.colorbar(mappable=im, ax=axs[j][i, :3], shrink=0.8, extend="max")
                    cbar.set_label(self.variables[i] + unit, fontsize=14)

                    # Plotting error sample sperately because of its different color scale
                    im_error = axs[j][i, 3].pcolormesh(lon, lat, error_sample, cmap=cmaps["error"], vmin=err_vmin, vmax=err_vmax, transform=platecarree_proj)

                    # Plotting the colorbar for the error
                    cbar_error = plt.colorbar(mappable=im_error, ax=axs[j][i, 3], shrink=0.8, extend="max")
                    cbar_error.set_label(self.variables[i] + unit, fontsize=14)

                else:

                    cmap = cmaps["temp"]
                    unit = " (°C)"

                    # Converting units in °C and computing scaling values for diverging colormap
                    if self.variables[i] == "tasmin":
                        lr_sample, hr_pred_sample, hr_sample = KToC(lrinterp[j,i]), KToC(hr_pred[j,i]), KToC(hr[j,i])
                    elif self.variables[i] == "tasmax":
                        if self.transfo:
                            lr_sample = KToC(softplus(lrinterp[j,i], c=0.) + lrinterp[j,i-1])
                            hr_pred_sample = KToC(softplus(hr_pred[j,i], c=0.) + hr_pred[j,i-1])
                            hr_sample = KToC(softplus(hr[j,i], c=0.) + hr[j,i-1])
                        else:
                            lr_sample = KToC(lrinterp[j,i])
                            hr_pred_sample = KToC(hr_pred[j,i])
                            hr_sample = KToC(hr[j,i])
                    max_abs = max(torch.amax(torch.abs(lr_sample)), torch.amax(torch.amax(hr_pred_sample)), torch.amax(torch.amax(hr_sample)))
                    vmin, vmax = -max_abs, max_abs

                    # Storing max_abs for computing shared vmin and vmax values for tasmin and tasmax later
                    temp_max_abs.append(max_abs)

                    # Computing absolute error and setting corresponding vmin, vmax
                    error_sample = torch.abs(hr_sample - hr_pred_sample)
                    err_vmin, err_vmax = 0, torch.amax(error_sample)

                    # Setting cartopy features on the Axes objects
                    for l in range(4):
                        axs[j][i, l].coastlines()
                        gl = axs[j][i, l].gridlines(crs=platecarree_proj, draw_labels=True, x_inline=False, y_inline=False, linestyle="--")
                        gl.top_labels = False
                        gl.right_labels = False
                        if l > 0:
                            gl.left_labels = False

                    # Plotting samlpes in the following order: upsampled low-resolution, predicted high-resolution, groundtruth high-resolution
                    im1 = axs[j][i, 0].pcolormesh(lon, lat, lr_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)
                    im2 = axs[j][i, 1].pcolormesh(lon, lat, hr_pred_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)
                    im3 = axs[j][i, 2].pcolormesh(lon, lat, hr_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)

                    # Plotting the colorbar for the row with the correct label
                    cbar = plt.colorbar(mappable=im3, ax=axs[j][i, :3], shrink=0.8, extend="both")
                    cbar.set_label(self.variables[i] + unit, fontsize=14)

                    temp_ims.append([im1, im2, im3])

                    # Plotting error sample sperately because of its different color scale
                    im_error = axs[j][i, 3].pcolormesh(lon, lat, error_sample, cmap=cmaps["error"], vmin=err_vmin, vmax=err_vmax, transform=platecarree_proj)

                    # Plotting the colorbar for the error
                    cbar_error = plt.colorbar(mappable=im_error, ax=axs[j][i, 3], shrink=0.8, extend="max")
                    cbar_error.set_label(self.variables[i] + unit, fontsize=14)

            shared_max_abs = np.max(temp_max_abs)
            for im in temp_ims:
                for im_c in im:
                    im_c.set_clim(vmin=-shared_max_abs, vmax=shared_max_abs)

            subfigs[j].suptitle(str(float_to_date(timestamps[j]))[:10], fontsize=16)

            axs[j][0, 0].set_title("Low-resolution", fontsize=14)
            axs[j][0, 1].set_title("Prediction", fontsize=14)
            axs[j][0, 2].set_title("High-resolution", fontsize=14)
            axs[j][0, 3].set_title("Absolute error", fontsize=14)

        fig.suptitle("Predictions after the " + str(epoch) + "th epoch for " + str(N) + " random validation dates", fontsize=18, fontweight='bold')

        plt.show()

        return fig, axs


    def plot_sample_batch(self, lrinterp, hr_preds, hr, timestamps_float, epoch, N=2, num_samples=3):

        """
        Plots low-resolution inputs, multiple high-resolution predictions, and ground truth high-resolution outputs.

        Parameters:
        - lrinterp (torch.Tensor): Interpolated low-resolution inputs of shape [N, nvars, H, W].
        - hr_preds (torch.Tensor): Predicted high-resolution outputs of shape [N, num_samples, nvars, H, W].
        - hr (torch.Tensor): Ground truth high-resolution outputs of shape [N, nvars, H, W].
        - timestamps (torch.Tensor): Timestamps corresponding to each sample.
        - epoch (int): Current epoch number, used for plot titles.
        - N (int): Number of samples to plot (default is 2).
        - num_samples (int): Number of high-resolution predictions per input (default is 3).

        Returns:
        - fig: The matplotlib figure object containing the plots.
        - axs: The axes of the plots for further customization if needed.
        """

        # Initialize projections
        rotatedpole_prj = ccrs.RotatedPole(pole_longitude=83.0, pole_latitude=42.5)
        platecarree_proj = ccrs.PlateCarree()

        # Total columns: low-res input + predicted high-res (num_samples) + ground truth high-res
        total_cols = num_samples + 2  # 1 (low-res) + num_samples (predictions) + 1 (ground truth)

        # Initialize figure and subfigures
        fig = plt.figure(figsize=(total_cols * 6, N * self.nvars * 4), constrained_layout=True)
        subfigs = fig.subfigures(N, 1, hspace=0.1)

        # Ensure subfigs is iterable
        if N == 1:
            subfigs = [subfigs]

        # Define colormaps
        prep_colors = [
            (1., 1., 1.), 
            (0.5, 0.88, 1.),
            (0.1, 0.15, 0.8),
            (0.39, 0.09, 0.66), 
            (0.85, 0.36, 0.14),
            (0.99, 0.91, 0.3)
        ]
        prep_colormap = mpl.colors.LinearSegmentedColormap.from_list(name="prep", colors=prep_colors)
        cmaps = {'pr': prep_colormap, 'temp': cm.get_cmap('RdBu_r')}

        # Loop over each low-resolution input
        for j in range(N):
            # Create subplots for each variable and column
            axs = subfigs[j].subplots(self.nvars, total_cols, subplot_kw={'projection': rotatedpole_prj})

            # Ensure axs is a 2D array
            if self.nvars == 1:
                axs = np.array([axs])
            elif self.nvars > 1 and total_cols == 1:
                axs = axs[:, np.newaxis]
            else:
                axs = np.array(axs)
            

            # Extract latitude and longitude data
            lat = self.lat.sel(time=str(float_to_date(timestamps_float[j].item()))[:10]).load().to_numpy().squeeze()
            lon = self.lon.sel(time=str(float_to_date(timestamps_float[j].item()))[:10]).load().to_numpy().squeeze()

            temp_max_abs = []
            temp_ims = []
            # Loop over variables
            for i in range(self.nvars):
                var = self.variables[i]

                if var == "pr":
                    cmap = cmaps["pr"]
                    unit = " (mm/day)"
                    
                    if self.transfo:
                        lr_sample = kgm2sTommday(softplus(lrinterp[j, i]))
                        hr_sample = kgm2sTommday(softplus(hr[j, i]))
                        hr_pred_samples = [kgm2sTommday(softplus(hr_preds[j, s, i])) for s in range(num_samples)]
                    else:
                        lr_sample = kgm2sTommday(lrinterp[j, i])
                        hr_sample = kgm2sTommday(hr[j, i])
                        hr_pred_samples = [kgm2sTommday(hr_preds[j, s, i]) for s in range(num_samples)]


                    vmin = 0
                    vmax = max(torch.amax(lr_sample), torch.amax(hr_sample), max(torch.amax(pred) for pred in hr_pred_samples))

                    # Plot low-resolution input
                    axs[i, 0].pcolormesh(lon, lat, lr_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)
                    axs[i, 0].set_title("Low-resolution", fontsize=14)
                    # Plot predictions
                    for s in range(num_samples):
                        axs[i, s+1].pcolormesh(lon, lat, hr_pred_samples[s], cmap=cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)
                        axs[i, s+1].set_title(f"Prediction {s+1}", fontsize=14)
                    # Plot ground truth
                    im = axs[i, -1].pcolormesh(lon, lat, hr_sample, cmap=cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)
                    axs[i, -1].set_title("High-resolution", fontsize=14)
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=axs[i, :], orientation='vertical', shrink=0.8, extend="max")
                    cbar.set_label(var + unit, fontsize=14)

                else:  # 'tasmin' or 'tasmax'
                    cmap = cmaps["temp"]
                    unit = " (°C)"
                    if self.variables[i] == "tasmin":
                        lr_sample = KToC(lrinterp[j, i])
                        hr_sample = KToC(hr[j, i])
                        hr_pred_samples = [KToC(hr_preds[j, s, i]) for s in range(num_samples)]
                    elif self.variables[i] == "tasmax":
                        if self.transfo:
                            lr_sample = KToC(softplus(lrinterp[j, i], c=0.) + lrinterp[j, i-1])
                            hr_sample = KToC(softplus(hr[j, i], c=0.) + hr[j, i-1])
                            hr_pred_samples = [KToC(softplus(hr_preds[j, s, i], c=0.) + hr_preds[j, s, i-1]) for s in range(num_samples)]
                        else:
                            lr_sample = KToC(lrinterp[j, i])
                            hr_sample = KToC(hr[j, i])
                            hr_pred_samples = [KToC(hr_preds[j, s, i]) for s in range(num_samples)]

                    max_abs = max(torch.amax(torch.abs(lr_sample)), torch.amax(torch.abs(hr_sample)),
                                max(torch.amax(torch.abs(pred)) for pred in hr_pred_samples))
                    
                    vmin = -max_abs
                    vmax = max_abs
                    
                    temp_max_abs.append(max_abs)

                    # Ensure max_abs is positive
                    if max_abs <= 0 or np.isnan(max_abs):
                        print(f"Invalid max_abs value: {max_abs}")
                        max_abs = 1  # Set to a default positive value to avoid errors

                    # Plot low-resolution input
                    axs[i, 0].pcolormesh(lon, lat, lr_sample, cmap=cmap, vmin=vmin, vmax=max_abs, transform=platecarree_proj)
                    axs[i, 0].set_title("Low-resolution", fontsize=14)
                    # Plot predictions
                    for s in range(num_samples):
                        axs[i, s+1].pcolormesh(lon, lat, hr_pred_samples[s], cmap=cmap, vmin=vmin, vmax=max_abs, transform=platecarree_proj)
                        axs[i, s+1].set_title(f"Prediction {s+1}", fontsize=14)
                    # Plot ground truth
                    im = axs[i, -1].pcolormesh(lon, lat, hr_sample, cmap=cmap, vmin=vmin, vmax=max_abs, transform=platecarree_proj)
                    axs[i, -1].set_title("High-resolution", fontsize=14)
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=axs[i, :], orientation='vertical', shrink=0.8, extend="both")
                    cbar.set_label(var + unit, fontsize=14)
                    temp_ims.append(im)

                # Add coastlines and gridlines
                for l in range(total_cols):
                    axs[i, l].coastlines()
                    gl = axs[i, l].gridlines(crs=platecarree_proj, draw_labels=True, x_inline=False, y_inline=False, linestyle="--")
                    gl.top_labels = False
                    gl.right_labels = False
                    if l > 0:
                        gl.left_labels = False

            # Adjust temperature colormaps to have shared limits
            if temp_max_abs:
                shared_max_abs = np.max(temp_max_abs)
                for im in temp_ims:
                    im.set_clim(vmin=-shared_max_abs, vmax=shared_max_abs)

            # Set row titles
            for i in range(self.nvars):
                axs[i, 0].set_ylabel(f"{self.variables[i]}", fontsize=14)

            # Set supertitle for the sample
            subfigs[j].suptitle(f"Sample {j+1}: {str(float_to_date(timestamps_float[j].item()))[:10]}", fontsize=16)

        # Set overall figure title
        fig.suptitle(f"Predictions after the {epoch}th epoch", fontsize=18, fontweight='bold')

        return fig, axs

    def plot_residual_sample_batch(self, lrinterp, residual_preds, hr, timestamps_float, epoch, N=2, num_samples=3):
        """
        Plots:
        - Low-resolution interpolated inputs (first column),
        - Multiple residual predictions (middle columns),
        - High-resolution ground truth outputs (last column).

        Parameters:
            lrinterp (torch.Tensor): Interpolated low-resolution inputs of shape [N, nvars, H, W].
            residual_preds (torch.Tensor): Predicted residuals of shape [N, num_samples, nvars, H, W].
            hr (torch.Tensor): Ground truth high-resolution outputs of shape [N, nvars, H, W].
            timestamps_float (torch.Tensor): Float timestamps corresponding to each sample.
            epoch (int): Current epoch number, used for plot titles.
            N (int): Number of samples to plot (default is 2).
            num_samples (int): Number of residual predictions per input (default is 3).

        Returns:
            fig, axs: Matplotlib figure and axes objects.
        """

        rotatedpole_prj = ccrs.RotatedPole(pole_longitude=83.0, pole_latitude=42.5)
        platecarree_proj = ccrs.PlateCarree()

        # Total columns: low-res input + predicted residuals (num_samples) + ground truth high-res
        total_cols = num_samples + 2

        fig = plt.figure(figsize=(total_cols * 6, N * self.nvars * 4), constrained_layout=True)
        subfigs = fig.subfigures(N, 1, hspace=0.1)
        if N == 1:
            subfigs = [subfigs]

        # Colormaps
        prep_colors = [
            (1., 1., 1.), 
            (0.5, 0.88, 1.),
            (0.1, 0.15, 0.8),
            (0.39, 0.09, 0.66), 
            (0.85, 0.36, 0.14),
            (0.99, 0.91, 0.3)
        ]
        prep_colormap = mpl.colors.LinearSegmentedColormap.from_list(name="prep", colors=prep_colors)
        temp_cmap = cm.get_cmap('RdBu_r')  # For temperature
        residual_cmap = cm.get_cmap('RdBu_r')  # Diverging colormap for residuals

        for j in range(N):
            axs = subfigs[j].subplots(self.nvars, total_cols, subplot_kw={'projection': rotatedpole_prj})

            if self.nvars == 1:
                axs = np.array([axs])
            elif self.nvars > 1 and total_cols == 1:
                axs = axs[:, np.newaxis]
            else:
                axs = np.array(axs)

            # Extract latitude and longitude data
            lat = self.lat.sel(time=str(float_to_date(timestamps_float[j].item()))[:10]).load().to_numpy().squeeze()
            lon = self.lon.sel(time=str(float_to_date(timestamps_float[j].item()))[:10]).load().to_numpy().squeeze()

            for i, var in enumerate(self.variables):
                # Transform lrinterp and hr for visualization (as in the original code)
                if var == "pr":
                    # Convert to mm/day
                    if self.transfo:
                        lr_sample = kgm2sTommday(softplus(lrinterp[j, i]))
                        hr_sample = kgm2sTommday(softplus(hr[j, i]))
                    else:
                        lr_sample = kgm2sTommday(lrinterp[j, i])
                        hr_sample = kgm2sTommday(hr[j, i])

                    # Residuals are dimensionless (standardized), so just plot as is.
                    # Determine vmin/vmax for residuals
                    residual_var = residual_preds[j, :, i, :, :]  # shape [num_samples, H, W]
                    max_abs_res = torch.max(torch.abs(residual_var))
                    vmin_res, vmax_res = -max_abs_res, max_abs_res

                    # Determine vmin/vmax for lr and hr
                    vmin = 0
                    vmax = max(torch.amax(lr_sample), torch.amax(hr_sample))

                    # Plot low-res
                    axs[i, 0].pcolormesh(lon, lat, lr_sample, cmap=prep_colormap, vmin=vmin, vmax=vmax, transform=platecarree_proj)
                    axs[i, 0].set_title("Low-resolution", fontsize=14)

                    # Plot each residual prediction
                    for s in range(num_samples):
                        axs[i, s+1].pcolormesh(
                            lon, lat, residual_preds[j, s, i], 
                            cmap=residual_cmap, vmin=vmin_res, vmax=vmax_res, transform=platecarree_proj
                        )
                        axs[i, s+1].set_title(f"Residual {s+1}", fontsize=14)

                    # Plot high-res
                    im = axs[i, -1].pcolormesh(lon, lat, hr_sample, cmap=prep_colormap, vmin=vmin, vmax=vmax, transform=platecarree_proj)
                    axs[i, -1].set_title("High-resolution", fontsize=14)

                    # Add colorbars
                    cbar_hr = plt.colorbar(im, ax=axs[i, :], orientation='vertical', shrink=0.8, extend="max")
                    cbar_hr.set_label(var + " (mm/day)", fontsize=14)

                    # Add a separate colorbar for residuals if desired (optional)
                    # If needed, we can just rely on the same colorbar
                    # but since residual columns might have different scales, we won't add another cbar for each residual column here.

                else:
                    # Temperature variables
                    if var == "tasmin":
                        lr_sample = KToC(lrinterp[j, i])
                        hr_sample = KToC(hr[j, i])
                    elif var == "tasmax":
                        if self.transfo:
                            lr_sample = KToC(softplus(lrinterp[j, i], c=0.) + lrinterp[j, i-1])
                            hr_sample = KToC(softplus(hr[j, i], c=0.) + hr[j, i-1])
                        else:
                            lr_sample = KToC(lrinterp[j, i])
                            hr_sample = KToC(hr[j, i])

                    # Determine vmin/vmax for hr/ lr
                    max_abs = max(torch.amax(torch.abs(lr_sample)), torch.amax(torch.abs(hr_sample)))
                    max_abs = max(max_abs, 1e-7)  # Avoid zero division
                    vmin, vmax = -max_abs, max_abs

                    # Residuals for this variable
                    residual_var = residual_preds[j, :, i, :, :]  # [num_samples, H, W]
                    max_abs_res = torch.max(torch.abs(residual_var))
                    max_abs_res = max(max_abs_res, 1e-7)
                    vmin_res, vmax_res = -max_abs_res, max_abs_res

                    # Plot low-res
                    axs[i, 0].pcolormesh(lon, lat, lr_sample, cmap=temp_cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)
                    axs[i, 0].set_title("Low-resolution", fontsize=14)

                    # Plot residual predictions
                    for s in range(num_samples):
                        axs[i, s+1].pcolormesh(
                            lon, lat, residual_preds[j, s, i],
                            cmap=residual_cmap, vmin=vmin_res, vmax=vmax_res, transform=platecarree_proj
                        )
                        axs[i, s+1].set_title(f"Residual {s+1}", fontsize=14)

                    # Plot high-res
                    im = axs[i, -1].pcolormesh(lon, lat, hr_sample, cmap=temp_cmap, vmin=vmin, vmax=vmax, transform=platecarree_proj)
                    axs[i, -1].set_title("High-resolution", fontsize=14)

                    # Add colorbar
                    cbar_hr = plt.colorbar(im, ax=axs[i, :], orientation='vertical', shrink=0.8, extend="both")
                    cbar_hr.set_label(var + " (°C)", fontsize=14)

                # Add coastlines and gridlines
                for col in range(total_cols):
                    axs[i, col].coastlines()
                    gl = axs[i, col].gridlines(crs=platecarree_proj, draw_labels=True, x_inline=False, y_inline=False, linestyle="--")
                    gl.top_labels = False
                    gl.right_labels = False
                    if col > 0:
                        gl.left_labels = False

                axs[i, 0].set_ylabel(f"{var}", fontsize=14)

            subfigs[j].suptitle(f"Sample {j+1}: {str(float_to_date(timestamps_float[j].item()))[:10]}", fontsize=16)

        fig.suptitle(f"Residual Predictions after the {epoch}th epoch", fontsize=18, fontweight='bold')

        return fig, axs
    
    def plot_residual_differences(self, residual_preds, timestamps_float, epoch, N=2, num_samples=3):
        """
        Plots the pixel-wise differences between residual predictions:
        Differences if num_samples=3:
        - Diff(1-2)
        - Diff(1-3)
        - Diff(2-3)
        These are plotted in a separate figure, one figure for all differences.

        Parameters:
            residual_preds (torch.Tensor): Residual predictions [N, num_samples, nvars, H, W]
            timestamps_float (torch.Tensor): Float timestamps for each sample.
            epoch (int): Current epoch number.
            N (int): Number of samples to plot.
            num_samples (int): Number of residual predictions per input (default=3).
        """

        if num_samples != 3:
            raise ValueError("This function is implemented for exactly 3 samples to compare differences.")

        # Differences:
        difference_pairs = [(0,1), (0,2), (1,2)]
        num_diff_maps = len(difference_pairs)  # should be 3

        rotatedpole_prj = ccrs.RotatedPole(pole_longitude=83.0, pole_latitude=42.5)
        platecarree_proj = ccrs.PlateCarree()

        # We'll have num_diff_maps columns (3 columns if num_samples=3)
        total_cols = num_diff_maps

        fig = plt.figure(figsize=(total_cols * 6, N * self.nvars * 4), constrained_layout=True)
        subfigs = fig.subfigures(N, 1, hspace=0.1)
        if N == 1:
            subfigs = [subfigs]

        residual_cmap = cm.get_cmap('RdBu_r')  # Diverging colormap for differences

        for j in range(N):
            axs = subfigs[j].subplots(self.nvars, total_cols, subplot_kw={'projection': rotatedpole_prj})

            if self.nvars == 1:
                axs = np.array([axs])
            elif self.nvars > 1 and total_cols == 1:
                axs = axs[:, np.newaxis]
            else:
                axs = np.array(axs)

            # Convert the dataset's calendar to proleptic_gregorian
            self.data = self.data.convert_calendar('proleptic_gregorian')

            # Update lat and lon after conversion
            self.lat = self.data['lat']
            self.lon = self.data['lon']

            date_str = str(float_to_date(timestamps_float[j].item()))[:10]  # "1994-03-12"
            lat = self.lat.sel(time=date_str, method="nearest").load().to_numpy().squeeze()
            lon = self.lon.sel(time=date_str, method="nearest").load().to_numpy().squeeze()

            # Compute differences for the j-th sample
            diff_maps = []
            for (a, b) in difference_pairs:
                diff_maps.append(residual_preds[j, b] - residual_preds[j, a])
            diff_maps = torch.stack(diff_maps, dim=0)  # [num_diff_maps, nvars, H, W]

            for i, var in enumerate(self.variables):
                # Determine scale for differences
                diff_max_abs = torch.max(torch.abs(diff_maps[:, i, :, :]))
                diff_max_abs = max(diff_max_abs, 1e-7)
                vmin_diff, vmax_diff = -diff_max_abs, diff_max_abs

                for d_idx, (a, b) in enumerate(difference_pairs):
                    im_diff = axs[i, d_idx].pcolormesh(
                        lon, lat, diff_maps[d_idx, i],
                        cmap=residual_cmap, vmin=vmin_diff, vmax=vmax_diff, transform=platecarree_proj
                    )
                    axs[i, d_idx].set_title(f"Diff {a+1}-{b+1}")
                    axs[i, d_idx].coastlines()
                    gl = axs[i, d_idx].gridlines(crs=platecarree_proj, draw_labels=True, x_inline=False, y_inline=False, linestyle="--")
                    gl.top_labels = False
                    gl.right_labels = False
                    if d_idx > 0:
                        gl.left_labels = False

                axs[i, 0].set_ylabel(f"{var}")
                # Add a colorbar for the differences at the end of each row
                cbar_diff = plt.colorbar(im_diff, ax=axs[i, :], orientation='vertical', shrink=0.8)
                cbar_diff.set_label("Difference", fontsize=14)

            subfigs[j].suptitle(f"Sample {j+1}: {str(float_to_date(timestamps_float[j].item()))[:10]}")

        fig.suptitle(f"Residual Prediction Differences after the {epoch}th epoch", fontsize=16)
        return fig, axs
  
