import numpy as np
import pandas as pd

# loguru
from loguru import logger


def single_motion(transrotdata, p_slomo_threshold, p_trans_rot_scale):
    # p_trans_rot_scale=100;                   # scale factor trans vs rot motion
    # p_slomo_threshold=5;                      #  threshold for motion signal

    motiondata = {}

    # translat speed
    motiondata["trans"] = np.linalg.norm(np.diff(transrotdata.iloc[:, :3], axis=0), axis=1)
    # rotat speed
    motiondata["rot"] = np.linalg.norm(np.diff(transrotdata.iloc[:, 3:6], axis=0), axis=1) * 180 / np.pi

    motiondata["thres_trans"] = p_slomo_threshold
    motiondata["thres_rot"] = p_slomo_threshold

    motiondata["trans_t"] = motiondata["trans"] * (motiondata["trans"] > motiondata["thres_trans"])
    motiondata["rot_t"] = motiondata["rot"] * (motiondata["rot"] > motiondata["thres_rot"])

    motiondata["both_not_normed"] = motiondata["trans_t"] + motiondata["rot_t"] * p_trans_rot_scale
    if np.max(motiondata["both_not_normed"]) != 0:
        motiondata["both"] = motiondata["both_not_normed"] / np.max(motiondata["both_not_normed"])
    else:
        motiondata["both"] = motiondata["both_not_normed"]

    motiondata["numvols"] = len(motiondata["both"])

    return motiondata


def moving_average(n_fmri, window_size):
    # Erstelle Gewichtungsmatrix
    weighting_matrix = np.zeros((n_fmri, n_fmri))
    # Schleifen zur Füllung der Gewichtungsmatrix
    for i in range(n_fmri):
        for j in range(window_size // 2 + 1):
            if i - j >= 0:
                weighting_matrix[i, i - j] = 1
            if i + j < n_fmri:
                weighting_matrix[i, i + j] = 1
    return weighting_matrix


def calc_weighted_matrix_by_realignment_parameters_file(rp_file, n_fmri, k, threshold=5):
    # Lade Realignment Parameter Datei
    # rp_data = pd.read_csv(rp_file, sep="\t", header=None, skiprows=1, decimal=".")
    rp_data = pd.read_csv(
        rp_file,
        sep=r"\s+",
        header=None,
        comment="#",
        engine="python",
    )

    rp_data = rp_data.apply(pd.to_numeric, errors="raise")

    motiondata = single_motion(rp_data, threshold, 0)
    motiondata["both_not_normed"] = np.concatenate(
        (
            np.zeros(len(rp_data) - len(motiondata["both_not_normed"])),
            motiondata["both_not_normed"],
        )
    )

    n_data = len(motiondata["both_not_normed"])

    # Beachte Dummy-Scans
    diff_n = n_fmri - n_data
    if diff_n < 0:
        logger.error(
            "Number of volumes in the realignment parameter file is larger than the number of volumes in the fMRI data. Please check the realignment parameter file."
        )
        # raise error
        raise ValueError(
            "Number of volumes in the realignment parameter file is larger than the number of volumes in the fMRI data. Please check the realignment parameter file."
        )

    motiondata["both_not_normed"] = np.concatenate((np.zeros(diff_n), motiondata["both_not_normed"]))

    if np.max(motiondata["both_not_normed"]) > 0:
        slid_win = np.zeros((n_fmri, k))
        distance = np.zeros(n_fmri)
        for jslide in range(n_fmri):
            distance[:jslide] = np.arange(jslide, 0, -1)
            distance[jslide:] = np.arange(1, n_fmri - jslide + 1)

            # Integriere Bewegungsdaten in die Gewichtung
            motion_scaling = k / np.min(motiondata["both_not_normed"][motiondata["both_not_normed"] > 0])
            motion_effect = np.cumsum(
                np.concatenate(
                    (
                        -motiondata["both_not_normed"][:jslide],
                        motiondata["both_not_normed"][jslide:],
                    )
                )
            )
            distance += motion_scaling * motion_effect
            distance[motiondata["both_not_normed"] > 0] = np.nan

            distance -= np.nanmin(distance)
            sort_order = np.argsort(distance)[:k]
            slid_win[jslide, :] = sort_order

        # Erstelle Gewichtungsmatrix
        weighting_matrix = np.zeros((n_fmri, n_fmri))
        for j in range(n_fmri):
            weighting_matrix[j, slid_win[j, :].astype(int)] = 1
    else:
        weighting_matrix = moving_average(n_fmri, k)

    motiondata_struct = motiondata

    return motiondata_struct, weighting_matrix
