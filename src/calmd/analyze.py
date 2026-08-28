"""

"""

from typing import Union, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import axes
import xarray as xr
from scipy.stats import wasserstein_distance

from calmd.database import MultiDimDb
from calmd._setupbase import MscuaSetup


class MsCuaAnalyzer:
    """Used to visualize the results of an MsCua calibration iteration."""
    def __init__(self, dbase: Union[MultiDimDb, xr.Dataset]):

        if isinstance(dbase, MultiDimDb):
            self.dbase = dbase
            if self.dbase.format == 'memory':
                self.ds = self.dbase.to_xarray()
            else:
                self.ds = self.dbase
        else:
            self.dbase = None
            self.ds = dbase

    def plot_pfactor(self, threshold: bool = False):
        ax = plt.axes()
        ax.hist(self.ds.pfactor.values)
        if threshold:
            ax.axvline(self.ds.pfactor_threshold.values, color='black', ls='--', label='Threshold')
            ax.legend()
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_xlabel("pfactor Value", fontsize=12)
        plt.show()

    def plot_rfactor(self):
        ax = plt.axes()
        ax.hist(self.ds.rfactor.values)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_xlabel("rfactor Value", fontsize=12)
        plt.show()

    def plot_refined_parameters(self, param: str, indx: int = 0, ax: Optional[axes.Axes] = None, **kwargs):
        if ax is None:
            ax = plt.axes()
        else:
            ax = ax
        ax.hist(self.ds[f"{param}_refined"].values[:, indx])
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_xlabel(f"Refined {param} Value", fontsize=12)
        plt.show()

    def compare_parameter_distributions(self, param: str, comp_dstb: Union[np.ndarray, list], indx: int = 0,
                                        ax: Optional[axes.Axes] = None, **kwargs):
        dims = self.ds[f"{param}_samples"].dims
        if ax is None:
            ax = plt.axes()
        else:
            ax = ax
        ax.hist(comp_dstb, label='comparison parameter values', **kwargs)
        ax.hist(self.ds[f"{param}_refined"].values[:, indx], label='refined parameter values', **kwargs)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_xlabel(f"{param} Value", fontsize=12)
        ax.set_title(f"{dims[1]}: {indx}", fontsize=12)
        ax.legend()
        plt.show()

    def plot_ppu(self, indx: int = 0, best_sim: bool = False, obs_data: Optional[np.ndarray] = None,
                 series: Optional[str] = None, sort: bool = False, sort_low: bool = True,
                 ax: Optional[axes.Axes] = None):
        """Plots 95% Prediction Uncertainty for one model with all related obj funcs.

        Args:
            indx: which model number to display results for.
            best_sim: whether to plot the single best simulation for each objective function.
            obs_data: if given, plot observation data.
            series: if obs_data==True, specifies the name of the time series.
            sort: whether to sort values in ascending order. This can be useful to check that
              the model is producing a reasonable range of values independent of time.
            sort_low: if sort==True, this determines whether to sort the 95PPU values by the
              lower bound (True) or the upper bound (False).
            ax: adds plots to an existing axes object?
        """
        dims = self.ds["95PPU_lower"].dims
        if ax is None:
            ax = plt.axes()
        else:
            ax = ax
        if sort:
            # Remove any potential nan values so the sorted values line up.
            nonan_ind = np.where(~np.isnan(obs_data[:, indx]))
            if sort_low:
                srt_idx = np.argsort(self.ds["95PPU_lower"].values[nonan_ind, indx])
            else:
                srt_idx = np.argsort(self.ds["95PPU_upper"].values[nonan_ind, indx])

            ax.fill_between(np.arange(self.ds["95PPU_lower"].values.shape[0]),
                            self.ds["95PPU_upper"].values[srt_idx, indx][0],
                            self.ds["95PPU_lower"].values[srt_idx, indx][0],
                            label="95 Percentile Prediction Uncertainty",
                            ec='blue',
                            alpha=0.3)
            if obs_data is not None:
                if series:
                    label = f'Observed Values - {series}'
                else:
                    label = 'Observed Values'
                if self.ds["95PPU_lower"].values.shape != obs_data.shape:
                    raise ValueError(
                        "The input observation data does not match the size of the modeled 95 percent prediction uncertainty.")
                ax.plot(np.sort(obs_data[nonan_ind, indx][0]), marker=None, ls='-', lw=1.2, color='black',
                        label=label)
            if best_sim:
                for i in self.ds["objective_functions"].values:
                    ax.plot(self.ds[f"best_simulation_{i}"].values[srt_idx, indx][0], marker=None, ls='--', lw=1,
                            color='red', label=f'Best Model Simulation - {i}')
        else:
            ax.fill_between(np.arange(self.ds["95PPU_lower"].values.shape[0]),
                            self.ds["95PPU_upper"].values[:, indx],
                            self.ds["95PPU_lower"].values[:, indx],
                            label="95 Percentile Prediction Uncertainty")
            if obs_data is not None:
                if series:
                    label = f'Observed Values - {series}'
                else:
                    label = 'Observed Values'
                if self.ds["95PPU_lower"].values.shape != obs_data.shape:
                    raise ValueError(
                        "The input observation data does not match the size of the modeled 95 percent prediction uncertainty.")
                ax.plot(obs_data[:, indx], marker=None, ls='-', lw=1.25, color='black', label=label)
            if best_sim:
                for i in self.ds["objective_functions"].values:
                    ax.plot(self.ds[f"best_simulation_{i}"].values[:, indx], marker=None, ls='--', lw=1, color='red',
                            label=f'Best Model Simulation - {i}')
        ax.set_ylabel("Model Output Values", fontsize=12)
        ax.set_xlabel(dims[0], fontsize=12)
        ax.set_title(f"{dims[1]}: {indx}", fontsize=12)
        ax.legend()
        plt.show()

    def plot_objective_functions(self, indx: int = 0, threshold: bool = False):
        n = len(self.ds.objective_functions)
        if n % 2 == 0:
            fig, axs = plt.subplots(int(n / 2), 2, sharex=False)
        else:
            fig, axs = plt.subplots(int(n / 2 + 1), 2, sharex=False)
        for i, v in enumerate(self.ds.objective_functions.values.tolist()):
            if i % 2 == 0:
                if len(axs.shape) > 1:
                    ax_i = axs[int(i / 2), 0]
                else:
                    ax_i = axs[0]
                ax_i.hist(self.ds[v].values[:, indx])
                if threshold:
                    thresh = self.ds.obj_func_thresholds.sel(objective_functions=v).values
                    ax_i.axvline(thresh, color='black', ls='--', label='Threshold')
                    ax_i.legend()
                ax_i.set_title(f"{self.ds[v].dims[1]} {indx}: {v}")
            else:
                if len(axs.shape) > 1:
                    ax_i = axs[int(i / 2), 1]
                else:
                    ax_i = axs[1]
                ax_i.hist(self.ds[v].values[:, indx])
                if threshold:
                    thresh = self.ds.obj_func_thresholds.sel(objective_functions=v).values
                    ax_i.axvline(thresh, color='black', ls='--', label='Threshold')
                    ax_i.legend()
                ax_i.set_title(f"{self.ds[v].dims[1]} {indx}: {v}")
        if n % 2 != 0:
            if len(axs.shape) > 1:
                for l in axs[int(n / 2) - 1, 1].get_xaxis().get_majorticklabels():
                    l.set_visible(True)
                fig.delaxes(axs[int(n / 2), 1])
                for ax in axs[-1, :]:
                    ax.set_xlabel('Objective Function Value')
                for ax in axs[:, 0]:
                    ax.set_ylabel('Frequency')
            else:
                for l in axs[1].get_xaxis().get_majorticklabels():
                    l.set_visible(True)
                fig.delaxes(axs[1])
                for ax in axs:
                    ax.set_xlabel('Objective Function Value')
                axs[0].set_ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    def plot_best_objfuncs(self, threshold: bool = False):
        n = len(self.ds.objective_functions)
        if n % 2 == 0:
            fig, axs = plt.subplots(int(n / 2), 2, sharex=False)
        else:
            fig, axs = plt.subplots(int(n / 2 + 1), 2, sharex=False)
        for i, v in enumerate(self.ds.objective_functions.values.tolist()):
            if i % 2 == 0:
                if len(axs.shape) > 1:
                    ax_i = axs[int(i / 2), 0]
                else:
                    ax_i = axs[0]
                ax_i.hist(self.ds.best_obj_function.sel(objective_functions=v).values)
                if threshold:
                    thresh = self.ds.obj_func_thresholds.sel(objective_functions=v).values
                    ax_i.axvline(thresh, color='black', ls='--', label='Threshold')
                    ax_i.legend()
                ax_i.set_title(f"Best {v} value per {self.ds[v].dims[1]}")
            else:
                if len(axs.shape) > 1:
                    ax_i = axs[int(i / 2), 1]
                else:
                    ax_i = axs[1]
                ax_i.hist(self.ds.best_obj_function.sel(objective_functions=v).values)
                if threshold:
                    thresh = self.ds.obj_func_thresholds.sel(objective_functions=v).values
                    ax_i.axvline(thresh, color='black', ls='--', label='Threshold')
                    ax_i.legend()
                ax_i.set_title(f"Best {v} value per {self.ds[v].dims[1]}")
        if n % 2 != 0:
            if len(axs.shape) > 1:
                for l in axs[int(n / 2) - 1, 1].get_xaxis().get_majorticklabels():
                    l.set_visible(True)
                fig.delaxes(axs[int(n / 2), 1])
                for ax in axs[-1, :]:
                    ax.set_xlabel('Objective Function Value')
                for ax in axs[:, 0]:
                    ax.set_ylabel('Frequency')
            else:
                for l in axs[1].get_xaxis().get_majorticklabels():
                    l.set_visible(True)
                fig.delaxes(axs[1])
                for ax in axs:
                    ax.set_xlabel('Objective Function Value')
                axs[0].set_ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    def plot_number_refined_params(self, threshold: bool = False):
        arb_parm = self.ds.parameters.values[0]
        nans = np.isnan(self.ds[f"{arb_parm}_refined"].values)
        nz = np.count_nonzero(~nans, axis=0)
        ax = plt.axes()
        ax.hist(nz)
        if threshold:
            ax.axvline(self.ds.min_refined_params_threshold.values, color='black', ls='--', label='Threshold')
            ax.legend()
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_xlabel("Number Refined Parameter Sets", fontsize=12)
        plt.show()

    def calc_param_distribution_change(self, param: str, comp_dstb: Union[np.ndarray, list], indx: int = 0):
        d1 = comp_dstb
        d2 = self.ds[f"{param}_refined"].values[:, indx]
        ws_dist = wasserstein_distance(d1[~np.isnan(d1)], d2[~np.isnan(d2)])

        return ws_dist

class MsCuaAnalyzerDict(MsCuaAnalyzer):
    """Used to visualize the results of an MsCua calibration iteration with multliple calibration time series.

    Only methods that require adjustments for dealing with nested dictionaries are overridden.
    """
    def plot_pfactor(self, threshold: bool = False):
        pfactors = []
        for i in list(self.ds.data_vars.keys()):
            if 'pfactor' in i and 'threshold' not in i:
                pfactors.append(i)
        for i in pfactors:
            ax = plt.axes()
            ax.hist(self.ds[i].values)
            if threshold:
                ax.axvline(self.ds.pfactor_threshold.values, color='black', ls='--', label='Threshold')
                ax.legend()
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_xlabel(f"{i} Value", fontsize=12)
            plt.show()

    def plot_rfactor(self):
        rfactors = []
        for i in list(self.ds.data_vars.keys()):
            if 'rfactor' in i:
                rfactors.append(i)
        for i in rfactors:
            ax = plt.axes()
            ax.hist(self.ds[i].values)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_xlabel(f"{i} Value", fontsize=12)
            plt.show()

    def plot_ppu(self, indx: int = 0, best_sim: bool = False, obs_data: Optional[np.ndarray] = None,
                 series: Optional[str] = None, sort: bool = False, sort_low: bool = True,
                 ax: Optional[axes.Axes] = None):
        """Plots 95% Prediction Uncertainty for one model and one obs time series with all related obj funcs.

        Since observation data is not saved to results Dataset, series must be used to specify which
        time series the observation data represents when there are multiple options.

        Args:
            indx: which model number to display results for.
            best_sim: whether to plot the single best simulation for each objective function.
            obs_data: if given, plot observation data.
            series: if obs_data==True and multiple observation time series were used for calibration,
              specifies the name of the time series
            sort: whether to sort values in ascending order. This can be useful to check that
              the model is producing a reasonable range of values independent of time.
            sort_low: if sort==True, this determines whether to sort the 95PPU values by the
              lower bound (True) or the upper bound (False).
            ax: adds plots to an existing axes object?
        """
        ppus = []
        for i in list(self.ds.data_vars.keys()):
            if '95PPU_lower' in i:
                ppus.append(i)

        if len(ppus) == 1:
            dims = self.ds["95PPU_lower"].dims
            if ax is None:
                ax = plt.axes()
            else:
                ax = ax
            if sort:
                # Remove any potential nan values so the sorted values line up.
                nonan_ind = np.where(~np.isnan(obs_data[:, indx]))
                if sort_low:
                    srt_idx = np.argsort(self.ds["95PPU_lower"].values[nonan_ind, indx])
                else:
                    srt_idx = np.argsort(self.ds["95PPU_upper"].values[nonan_ind, indx])

                ax.fill_between(np.arange(self.ds["95PPU_lower"].values.shape[0]),
                                self.ds["95PPU_upper"].values[srt_idx, indx][0],
                                self.ds["95PPU_lower"].values[srt_idx, indx][0],
                                label="95 Percentile Prediction Uncertainty",
                                ec='blue',
                                alpha=0.3)
                if obs_data is not None:
                    if self.ds["95PPU_lower"].values.shape != obs_data.shape:
                        raise ValueError(
                            "The input observation data does not match the size of the modeled 95 percent prediction uncertainty.")
                    ax.plot(np.sort(obs_data[nonan_ind, indx][0]), marker=None, ls='-', lw=1.2, color='black',
                            label='Observed Values')
                if best_sim:
                    for i in self.ds["objective_functions"].values:
                        ax.plot(self.ds[f"best_simulation_{i}"].values[srt_idx, indx][0], marker=None, ls='--', lw=1,
                                color='red', label=f'Best Model Simulation - {i}')
            else:
                ax.fill_between(np.arange(self.ds["95PPU_lower"].values.shape[0]),
                                self.ds["95PPU_upper"].values[:, indx],
                                self.ds["95PPU_lower"].values[:, indx],
                                label="95 Percentile Prediction Uncertainty")
                if obs_data is not None:
                    if self.ds["95PPU_lower"].values.shape != obs_data.shape:
                        raise ValueError(
                            "The input observation data does not match the size of the modeled 95 percent prediction uncertainty.")
                    ax.plot(obs_data[:, indx], marker=None, ls='-', lw=1.25, color='black', label='Observed Values')
                if best_sim:
                    for i in self.ds["objective_functions"].values:
                        ax.plot(self.ds[f"best_simulation_{i}"].values[:, indx], marker=None, ls='--', lw=1,
                                color='red',
                                label=f'Best Model Simulation - {i}')
            ax.set_ylabel("Model Output Values", fontsize=12)
            ax.set_xlabel(dims[0], fontsize=12)
            ax.set_title(f"{dims[1]}: {indx}", fontsize=12)
            ax.legend()
            plt.show()
        elif len(ppus) > 1 and not series:
            print("Multiple time series observations, please choose one to plot with kwarg 'series'.")
        else:
            dims = self.ds[f"95PPU_lower_{series}"].dims
            if ax is None:
                ax = plt.axes()
            else:
                ax = ax
            if sort:
                # Remove any potential nan values so the sorted values line up.
                nonan_ind = np.where(~np.isnan(obs_data[:, indx]))
                if sort_low:
                    srt_idx = np.argsort(self.ds[f"95PPU_lower_{series}"].values[nonan_ind, indx])
                else:
                    srt_idx = np.argsort(self.ds[f"95PPU_upper_{series}"].values[nonan_ind, indx])

                ax.fill_between(np.arange(self.ds[f"95PPU_lower_{series}"].values.shape[0]),
                                self.ds[f"95PPU_upper_{series}"].values[srt_idx, indx][0],
                                self.ds[f"95PPU_lower_{series}"].values[srt_idx, indx][0],
                                label="95 Percentile Prediction Uncertainty",
                                ec='blue',
                                alpha=0.3)
                if obs_data is not None:
                    if self.ds[f"95PPU_lower_{series}"].values.shape != obs_data.shape:
                        raise ValueError(
                            "The input observation data does not match the size of the modeled 95 percent prediction uncertainty.")
                    ax.plot(np.sort(obs_data[nonan_ind, indx][0]), marker=None, ls='-', lw=1.2, color='black',
                            label='Observed Values')
                if best_sim:
                    for i in self.ds["objective_functions"].values:
                        if f"best_simulation_{i}_{series}" in self.ds.data_vars:
                            ax.plot(self.ds[f"best_simulation_{i}_{series}"].values[srt_idx, indx][0], marker=None,
                                    ls='--', lw=1,
                                    color='red', label=f'Best Model Simulation - {i}')
            else:
                ax.fill_between(np.arange(self.ds[f"95PPU_lower_{series}"].values.shape[0]),
                                self.ds[f"95PPU_upper_{series}"].values[:, indx],
                                self.ds[f"95PPU_lower_{series}"].values[:, indx],
                                label="95 Percentile Prediction Uncertainty")
                if obs_data is not None:
                    if self.ds[f"95PPU_lower_{series}"].values.shape != obs_data.shape:
                        raise ValueError(
                            "The input observation data does not match the size of the modeled 95 percent prediction uncertainty.")
                    ax.plot(obs_data[:, indx], marker=None, ls='-', lw=1.25, color='black', label='Observed Values')
                if best_sim:
                    for i in self.ds["objective_functions"].values:
                        if f"best_simulation_{i}_{series}" in self.ds.data_vars:
                            ax.plot(self.ds[f"best_simulation_{i}_{series}"].values[:, indx], marker=None, ls='--',
                                    lw=1, color='red',
                                    label=f'Best Model Simulation - {i}')
            ax.set_ylabel("Model Output Values", fontsize=12)
            ax.set_xlabel(dims[0], fontsize=12)
            ax.set_title(f"{dims[1]}: {indx}", fontsize=12)
            ax.legend()
            plt.show()

    def plot_objective_functions(self, indx: int = 0, threshold: bool = False):
        series_list = []
        for i in list(self.ds.data_vars.keys()):
            if 'rfactor' in i:
                series_list.append(i.split("_", maxsplit=1)[-1])

        for ts in series_list:
            ts_obfns = []
            for ob in self.ds.objective_functions.values:
                if f'{ob}_{ts}' in self.ds.data_vars:
                    ts_obfns.append(str(ob))
            n = len(ts_obfns)
            var_thresh = f'obj_func_thresholds_{ts}'
            if n % 2 == 0:
                fig, axs = plt.subplots(int(n / 2), 2, sharex=False)
            else:
                fig, axs = plt.subplots(int(n / 2 + 1), 2, sharex=False)
            for i, v in enumerate(ts_obfns):
                ts_v = f'{v}_{ts}'
                if i % 2 == 0:
                    if len(axs.shape) > 1:
                        ax_i = axs[int(i / 2), 0]
                    else:
                        ax_i = axs[0]
                    ax_i.hist(self.ds[ts_v].values[:, indx])
                    if threshold:
                        thresh = self.ds[var_thresh].sel(objective_functions=v).values
                        ax_i.axvline(thresh, color='black', ls='--', label='Threshold')
                        ax_i.legend()
                    ax_i.set_title(f"{self.ds[ts_v].dims[1]} {indx}: {ts_v}")
                else:
                    if len(axs.shape) > 1:
                        ax_i = axs[int(i / 2), 1]
                    else:
                        ax_i = axs[1]
                    ax_i.hist(self.ds[ts_v].values[:, indx])
                    if threshold:
                        thresh = self.ds[var_thresh].sel(objective_functions=v).values
                        ax_i.axvline(thresh, color='black', ls='--', label='Threshold')
                        ax_i.legend()
                    ax_i.set_title(f"{self.ds[ts_v].dims[1]} {indx}: {ts_v}")
            if n % 2 != 0:
                if len(axs.shape) > 1:
                    for l in axs[int(n / 2) - 1, 1].get_xaxis().get_majorticklabels():
                        l.set_visible(True)
                    fig.delaxes(axs[int(n / 2), 1])
                    for ax in axs[-1, :]:
                        ax.set_xlabel('Objective Function Value')
                    for ax in axs[:, 0]:
                        ax.set_ylabel('Frequency')
                else:
                    for l in axs[1].get_xaxis().get_majorticklabels():
                        l.set_visible(True)
                    fig.delaxes(axs[1])
                    for ax in axs:
                        ax.set_xlabel('Objective Function Value')
                    axs[0].set_ylabel('Frequency')
            plt.tight_layout()
            plt.show()

    def plot_best_objfuncs(self, threshold: bool = False):
        series_list = []
        for i in list(self.ds.data_vars.keys()):
            if 'rfactor' in i:
                series_list.append(i.split("_", maxsplit=1)[-1])

        for ts in series_list:
            ts_obfns = []
            for ob in self.ds.objective_functions.values:
                if f'{ob}_{ts}' in self.ds.data_vars:
                    ts_obfns.append(str(ob))
            n = len(ts_obfns)
            var_thresh = f'obj_func_thresholds_{ts}'
            var_best = f'best_obj_function_{ts}'
            if n % 2 == 0:
                fig, axs = plt.subplots(int(n / 2), 2, sharex=False)
            else:
                fig, axs = plt.subplots(int(n / 2 + 1), 2, sharex=False)
            for i, v in enumerate(ts_obfns):
                ts_v = f'{v}_{ts}'
                print(ts_v)
                if i % 2 == 0:
                    if len(axs.shape) > 1:
                        ax_i = axs[int(i / 2), 0]
                    else:
                        ax_i = axs[0]
                    ax_i.hist(self.ds[var_best].sel(objective_functions=v).values)  # why is this calling all nans?
                    if threshold:
                        thresh = self.ds[var_thresh].sel(objective_functions=v).values
                        ax_i.axvline(thresh, color='black', ls='--', label='Threshold')
                        ax_i.legend()
                    ax_i.set_title(f"Best {ts_v} value per {self.ds[ts_v].dims[1]}")
                else:
                    if len(axs.shape) > 1:
                        ax_i = axs[int(i / 2), 1]
                    else:
                        ax_i = axs[1]
                    ax_i.hist(self.ds[var_best].sel(objective_functions=v).values)
                    if threshold:
                        thresh = self.ds[var_thresh].sel(objective_functions=v).values
                        ax_i.axvline(thresh, color='black', ls='--', label='Threshold')
                        ax_i.legend()
                    ax_i.set_title(f"Best {ts_v} value per {self.ds[ts_v].dims[1]}")
            if n % 2 != 0:
                if len(axs.shape) > 1:
                    for l in axs[int(n / 2) - 1, 1].get_xaxis().get_majorticklabels():
                        l.set_visible(True)
                    fig.delaxes(axs[int(n / 2), 1])
                    for ax in axs[-1, :]:
                        ax.set_xlabel('Objective Function Value')
                    for ax in axs[:, 0]:
                        ax.set_ylabel('Frequency')
                else:
                    for l in axs[1].get_xaxis().get_majorticklabels():
                        l.set_visible(True)
                    fig.delaxes(axs[1])
                    for ax in axs:
                        ax.set_xlabel('Objective Function Value')
                    axs[0].set_ylabel('Frequency')
            plt.tight_layout()
            plt.show()
