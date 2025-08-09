import torch
import numpy as np
from scipy import stats
import pysteps as ps


def crps_over_groundtruth(hr, preds):

    # preds shape: (num_timestamps, num_samples, 3, h, w)
    # hr shape: (num_timestamps, 3, h, w)

    preds = preds.numpy()
    hr = hr.numpy()

    crps_vals = {var: [] for var in ["pr", "tasmin", "tasmax"]}
    for i in range(hr.shape[0]):
        hr_sample = hr[i].squeeze()
        preds_sample = preds[i].squeeze()
        crps_vals["pr"].append(ps.verification.probscores.CRPS(preds_sample[:, 0], hr_sample[0]))
        crps_vals["tasmin"].append(ps.verification.probscores.CRPS(preds_sample[:, 1], hr_sample[1]))
        crps_vals["tasmax"].append(ps.verification.probscores.CRPS(preds_sample[:, 2], hr_sample[2]))

    crps_vals["pr"] = np.mean(np.array(crps_vals["pr"]), axis=0)
    crps_vals["tasmin"] = np.mean(np.array(crps_vals["tasmin"]), axis=0)
    crps_vals["tasmax"] = np.mean(np.array(crps_vals["tasmax"]), axis=0)

    return crps_vals