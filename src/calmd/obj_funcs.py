import pandas as pd
import numpy as np
from numba import njit, prange

obj_func_direction = {
    'mse': 'minimize',
    'rmse': 'minimize',
    'nrmse': 'minimize',
    'nse': 'maximize',
    'pbias': 'minimize'
}


def mse_md(observation: np.ndarray, simulation: np.ndarray, return_dict: bool = False, axis=0):
    if not observation.flags.C_CONTIGUOUS:
        print('observation must be C_CONTIGUOUS')
    if not simulation.flags.C_CONTIGUOUS:
        print('observation must be C_CONTIGUOUS')
    if observation.shape[axis] == simulation.shape[axis]:
        mse = np.nanmean((observation - simulation) ** 2, axis=axis)
        if return_dict:
            return {'mse': mse}
        else:
            return mse
    else:
        raise ValueError("evaluation and simulation data do not have the same length.")


def rmse_md(observation: np.ndarray, simulation: np.ndarray, return_dict: bool = False, axis=0):
    if observation.shape[axis] == simulation.shape[axis]:
        mse = mse_md(observation, simulation, axis=axis)
        rmse = np.sqrt(mse)
        if return_dict:
            return {'rmse': rmse}
        else:
            return rmse
    else:
        raise ValueError("evaluation and simulation data do not have the same length.")


def nrmse_md(observation: np.ndarray, simulation: np.ndarray, return_dict: bool = False, axis=0):
    if observation.shape[axis] == simulation.shape[axis]:
        nrmse = rmse_md(observation, simulation, axis=axis) / np.nanmean(observation, axis=axis)
        if return_dict:
            return {'nrmse': nrmse}
        else:
            return nrmse
    else:
        raise ValueError("evaluation and simulation data do not have the same length.")


def nse_md(observation: np.ndarray, simulation: np.ndarray, return_dict: bool = False, axis=0):
    if observation.shape[axis] == simulation.shape[axis]:
        mean_observed = np.nanmean(observation, axis=axis)
        # compute numerator and denominator
        numerator = np.nansum((observation - simulation) ** 2, axis=axis)
        denominator = np.nansum((observation - mean_observed) ** 2, axis=axis)
        # compute coefficient
        nse = 1 - (numerator / denominator)
        if return_dict:
            return {'nse': nse}
        else:
            return nse
    else:
        raise ValueError("evaluation and simulation data do not have the same length.")


def pbias_md(observation: np.ndarray, simulation: np.ndarray, return_dict: bool = False, axis=0):
    if observation.shape[axis] == simulation.shape[axis]:
        pbias = np.nansum(simulation - observation, axis=axis) / np.nansum(observation, axis=axis)
        if return_dict:
            return {'pbias': pbias}
        else:
            return pbias
    else:
        raise ValueError("evaluation and simulation data do not have the same length.")


# @njit()
# def jit_nse_md(observation: np.ndarray, simulation: np.ndarray, return_dict: bool = True, axis=1):
#     if observation.shape[axis] == simulation[0, :, :].shape[axis]:
#         nse = np.zeros((simulation.shape[0], simulation.shape[2]))
#         for i in prange(observation.shape[1]):
#             observation_arr = observation[:, i]
#             simulation_arr = simulation[:, :, i]
#             mean_observed = np.nanmean(observation_arr)
#             numerator = np.nan_to_num((observation_arr[None, :] - simulation_arr)**2.0).sum(axis=1)[:, None]
#             denominator = np.nan_to_num((observation_arr[None, :] - mean_observed)**2.0).sum(axis=1)[:, None]
#             feature_nse = 1.0 - (numerator/denominator)
#             nse[:, i] = feature_nse[:, 0]
#         if return_dict:
#             return {'nse':nse}
#         else:
#             raise ValueError("values must be returned as dictionary for jit functions.")
#     else:
#         raise ValueError("evaluation and simulation data do not have the same length.")