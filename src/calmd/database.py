from typing import Union, Optional
from pathlib import Path
import numpy as np
import xarray as xr
import zarr

class MultiDimDb:
    format = None
    def __init__(
            self,
            dbname: str,
            dbpath: Optional[Union[str, Path]] = None,
            dims: Optional[dict] = None,
            dim_names: Optional[dict] = None,
            dbappend: bool = False,
            save_sim: bool = False,
    ):

        self.dbname = dbname
        self.cwd = dbpath
        self.dims = dims
        self.dim_names = dim_names
        self.dbappend = dbappend
        self.save_sim = save_sim

        self.ppu_upper = None
        self.ppu_lower = None
        self.pfactor = None
        self.rfactor = None
        self.thresholds = None
        self.best_sim = None
        self.best_params = None
        self.best_objfun = None


class ZarrDb(MultiDimDb):
    format = 'zarr'
    def __init__(self, *args, chunks: Optional[dict] = None, **kwargs):
        super(ZarrDb, self).__init__(*args, **kwargs)
        print("Zarr database initialized...")
        self.chunks = chunks
        self._par_samples = {}
        self._ref_par = {}
        self._objfun = {}
        self._sim = None
        self._obs_index = None

    @property
    def simulation_results(self):
        return self._sim

    @property
    def parameter_samples(self):
        return self._par_samples

    @property
    def refined_parameters(self):
        return self._ref_par

    @property
    def objective_func_values(self):
        return self._objfun

    def save(self, param_dict: Optional[dict] = None, objective_func: Optional[dict] = None,
             simulations: Optional[np.ndarray] = None, rep_ind: int = None):
        if (param_dict is None) & (objective_func is None) & (simulations is None):
            print("No data to save")
        else:
            if param_dict is not None:
                if not self._par_samples:
                    self._par_samples.update(param_dict)
                else:
                    for k, v in param_dict.items():
                        self._par_samples[k] = np.vstack((self._par_samples[k], v))
            if objective_func is not None:
                if not self._objfun:
                    self._objfun.update(objective_func)
                else:
                    for k, v in objective_func.items():
                        self._objfun[k] = np.vstack((self._objfun[k], v))
            if (self.parameter_samples is not None) & (simulations is not None) & (rep_ind is not None):
                if self._sim is not None:
                    self._sim[rep_ind, :, :] = simulations
                else:
                    itst = list(self._par_samples.keys())[0]
                    reps = self._par_samples[itst].shape[0]
                    # self._sim = zarr.open(self.cwd, mode='w', shape=((reps,) + simulations.shape),
                    #                       chunks=((1,) + simulations.shape), dtype=np.float32)
                    self._sim = zarr.open(self.cwd, mode='w', shape=((reps,) + simulations.shape),
                                          chunks=(1,) + (simulations.shape[0], 1), dtype=np.float32)
                    self._sim[rep_ind, :, :] = simulations

    def to_xarray(self):
        ds = xr.Dataset()
        param_names = []
        for k in list(self._par_samples.keys()):
            param_names.append(k)
        objfunc_names = []
        for k in list(self.objective_func_values):
            objfunc_names.append(k)
        if self.parameter_samples is not None:
            for pnm in param_names:
                ds[f"{pnm}_samples"] = (("repetition", self.dim_names[pnm]), self.parameter_samples[pnm],
                                        {"description": "original parameter sample"})
        if self.refined_parameters is not None:
            for pnm in param_names:
                ds[f"{pnm}_refined"] = (("repetition", self.dim_names[pnm]), self.refined_parameters[pnm],
                                        {"description": "refined parameter set"})
        if self.objective_func_values is not None:
            for obf in objfunc_names:
                ds[obf] = (("repetition", self.dim_names[list(self.dim_names.keys())[0]]),
                           self.objective_func_values[obf], {"description": "objective function"})
                if self.best_sim is not None:
                    ds[f"best_simulation_{obf}"] = (("observation", self.dim_names[list(self.dim_names.keys())[0]]),
                                                    self.best_sim[obf], {
                                                        "description": f"the simulation repetition with the best {obf} value"})

            if self.best_objfun is not None:
                bst_obfn = []
                bst_params = []
                for obf in objfunc_names:
                    bst_obfn.append(self.best_objfun[obf])
                    bst_p_mid = []
                    if self.best_params is not None:
                        for pnm in param_names:
                            bst_p_mid.append(self.best_params[obf][pnm])
                        bst_p_arr = np.array(bst_p_mid)
                        bst_params.append(bst_p_arr)
                ds["best_obj_function"] = (("objective_functions", self.dim_names[list(self.dim_names.keys())[0]]),
                                           np.array(bst_obfn))
                if self.best_params is not None:
                    ds["best_parameter_set"] = (
                        ("objective_functions", "parameters", self.dim_names[list(self.dim_names.keys())[0]]),
                        np.array(bst_params))

        if self.pfactor is not None:
            ds["pfactor"] = ((self.dim_names[list(self.dim_names.keys())[0]]), self.pfactor)
        if self.rfactor is not None:
            ds["rfactor"] = ((self.dim_names[list(self.dim_names.keys())[0]]), self.rfactor)
        if self.ppu_lower is not None:
            ds["95PPU_lower"] = (("observation", self.dim_names[list(self.dim_names.keys())[0]]), self.ppu_lower)
        if self.ppu_upper is not None:
            ds["95PPU_upper"] = (("observation", self.dim_names[list(self.dim_names.keys())[0]]), self.ppu_upper)

        coord_dict = {}
        for k, v in ds.dims.items():
            if k == 'objective_functions':
                coord_dict[k] = (k, objfunc_names)
            elif k == 'parameters':
                coord_dict[k] = (k, param_names)
            else:
                coord_dict[k] = (k, np.arange(v) + 1)

        ds = ds.assign_coords(coord_dict)

        if self.thresholds is not None:
            obfthrs = np.empty(len(objfunc_names))
            for k, v in self.thresholds.items():
                if k == 'pfactor_threshold':
                    ds[k] = (('scalar',), np.array([v]))
                elif k == 'min_refined_params_threshold':
                    ds[k] = (('scalar',), np.array([v]))
                else:
                    if k in objfunc_names:
                        obidx = objfunc_names.index(k)
                        obfthrs[obidx] = v
            if not np.isinf(obfthrs).all():
                ds['obj_func_thresholds'] = (('objective_functions',), obfthrs)

        return ds

    # don't need properties...everything will be a datavar in an xarray dataset, could have a property and setter
    # for the xarray.DataTree

    # @staticmethod
    # def load(dbpth: Union[str, Path]):
    #     pass


class MemDb(MultiDimDb):
    format = 'memory'

    def __init__(self, *args, **kwargs):

        super(MemDb, self).__init__(*args, dbpath=None, **kwargs)
        print("In Memory database initialized...")
        self._par_samples = {}
        self._ref_par = {}
        self._objfun = {}
        self._sim = []
        self._obs_index = None

    @property
    def simulation_results(self):
        return np.array(self._sim)

    @property
    def parameter_samples(self):
        return self._par_samples

    @property
    def refined_parameters(self):
        return self._ref_par

    @property
    def objective_func_values(self):
        return self._objfun

    @property
    def observation_labels(self):
        return self._obs_index

    @observation_labels.setter
    def observation_labels(self, new_in: np.ndarray):
        if self.simulation_results is None:
            raise AttributeError("No simulation results, save simulations first before assigning observation labels.")
        if new_in.size != self.simulation_results[0].shape[1]:
            raise ValueError("The input array is not the same length as observations dimension.")
        self._obs_index = new_in

    def save(self, param_dict: Optional[dict] = None, objective_func: Optional[dict] = None,
             simulations: Optional[np.ndarray] = None):
        if (param_dict is None) & (objective_func is None) & (simulations is None):
            print("No data to save")
        else:
            if param_dict is not None:
                if not self._par_samples:
                    self._par_samples.update(param_dict)
                else:
                    for k, v in param_dict.items():
                        self._par_samples[k] = np.vstack((self._par_samples[k], v))

            if objective_func is not None:
                if not self._objfun:
                    self._objfun.update(objective_func)
                else:
                    for k, v in objective_func.items():
                        self._objfun[k] = np.vstack((self._objfun[k], v))

            if simulations is not None:
                self._sim.append(simulations)

    def to_xarray(self):
        """To be used if there is a single calibration time series."""
        ds = xr.Dataset()
        param_names = []
        for k in list(self._par_samples.keys()):
            param_names.append(k)
        objfunc_names = []
        for k in list(self.objective_func_values):
            objfunc_names.append(k)
        if self.parameter_samples is not None:
            for pnm in param_names:
                ds[f"{pnm}_samples"] = (("repetition", self.dim_names[pnm]), self.parameter_samples[pnm],
                                        {"description": "original parameter sample"})
        if self.refined_parameters is not None:
            for pnm in param_names:
                ds[f"{pnm}_refined"] = (("repetition", self.dim_names[pnm]), self.refined_parameters[pnm],
                                        {"description": "refined parameter set"})
        if self.objective_func_values is not None:
            for obf in objfunc_names:
                ds[obf] = (
                    ("repetition", self.dim_names[list(self.dim_names.keys())[0]]), self.objective_func_values[obf],
                    {"description": "objective function"})
                if self.best_sim is not None:
                    ds[f"best_simulation_{obf}"] = (
                        ("observation", self.dim_names[list(self.dim_names.keys())[0]]), self.best_sim[obf],
                        {"description": f"the simulation repitition with the best {obf} value"})

            if self.best_objfun is not None:
                bst_obfn = []
                bst_params = []
                for obf in objfunc_names:
                    bst_obfn.append(self.best_objfun[obf])
                    bst_p_mid = []
                    if self.best_params is not None:
                        for pnm in param_names:
                            bst_p_mid.append(self.best_params[obf][pnm])
                        bst_p_arr = np.array(bst_p_mid)
                        bst_params.append(bst_p_arr)
                ds["best_obj_function"] = (
                    ("objective_functions", self.dim_names[list(self.dim_names.keys())[0]]), np.array(bst_obfn))
                if self.best_params is not None:
                    ds["best_parameter_set"] = (
                        ("objective_functions", "parameters", self.dim_names[list(self.dim_names.keys())[0]]),
                        np.array(bst_params))

        if len(self.simulation_results) != 0:
            ds["simulation_results"] = (
                ("repetition", "observation", self.dim_names[list(self.dim_names.keys())[0]]), np.array(self._sim))
        if self.pfactor is not None:
            ds["pfactor"] = ((self.dim_names[list(self.dim_names.keys())[0]]), self.pfactor)
        if self.rfactor is not None:
            ds["rfactor"] = ((self.dim_names[list(self.dim_names.keys())[0]]), self.rfactor)
        if self.ppu_lower is not None:
            ds["95PPU_lower"] = (("observation", self.dim_names[list(self.dim_names.keys())[0]]), self.ppu_lower)
        if self.ppu_upper is not None:
            ds["95PPU_upper"] = (("observation", self.dim_names[list(self.dim_names.keys())[0]]), self.ppu_upper)

        coord_dict = {}
        for k, v in ds.sizes.items():
            if k == 'objective_functions':
                coord_dict[k] = (k, objfunc_names)
            elif k == 'parameters':
                coord_dict[k] = (k, param_names)
            else:
                coord_dict[k] = (k, np.arange(v) + 1)

        ds = ds.assign_coords(coord_dict)

        if self.thresholds is not None:
            obfthrs = np.empty(len(objfunc_names))
            for k, v in self.thresholds.items():
                if k == 'pfactor_threshold':
                    ds[k] = (('scalar',), np.array([v]))
                elif k == 'min_refined_params_threshold':
                    ds[k] = (('scalar',), np.array([v]))
                else:
                    if k in objfunc_names:
                        obidx = objfunc_names.index(k)
                        obfthrs[obidx] = v
            if not np.isinf(obfthrs).all():
                ds['obj_func_thresholds'] = (('objective_functions',), obfthrs)

        return ds

    def to_xarray_dict(self):
        """To be used with multiple calibration time series."""
        ds = xr.Dataset()
        param_names = []
        for k in list(self._par_samples.keys()):
            param_names.append(k)

        # dict like {obs_time_series_1: [obj_func_name_1], obs_time_series_2: [obj_func_name_1, obj_func_name_2, ...], ...}
        objfunc_names = {}
        for k, v in self.objective_func_values.items():
            objfunc_names[k] = list(v.keys())

        # list like [obj_func_name_1, obj_func_name_2, ...]
        objfunc_names_flat = list({obj for objs in self.objective_func_values.values() for obj in objs.keys()})

        if self.parameter_samples is not None:
            for pnm in param_names:
                ds[f"{pnm}_samples"] = (("repetition", self.dim_names[pnm]), self.parameter_samples[pnm],
                                        {"description": "original parameter sample"})
        if self.refined_parameters is not None:
            for pnm in param_names:
                ds[f"{pnm}_refined"] = (("repetition", self.dim_names[pnm]), self.refined_parameters[pnm],
                                        {"description": "refined parameter set"})

        for ts, obfs in objfunc_names.items():
            if self.objective_func_values is not None:
                for obf in obfs:
                    ts_ob = f"{obf}_{ts}"
                    ds[ts_ob] = (
                        ("repetition", self.dim_names[list(self.dim_names.keys())[0]]),
                        self.objective_func_values[ts][obf],
                        {"description": "objective function"})
                    if self.best_sim is not None:
                        ds[f"best_simulation_{ts_ob}"] = (
                            (f"observation_{ts}", self.dim_names[list(self.dim_names.keys())[0]]),
                            self.best_sim[ts][obf],
                            {"description": f"the simulation repetition with the best {obf} value"})
                ts_len = len(self.objective_func_values[ts][obf][0])

                if self.best_objfun is not None:
                    bst_obfn = []
                    bst_params = []
                    for obf in objfunc_names_flat:
                        if obf in obfs:
                            bst_obfn.append(self.best_objfun[ts][obf])
                            bst_p_mid = []
                            if self.best_params is not None:
                                for pnm in param_names:
                                    bst_p_mid.append(self.best_params[ts][obf][pnm])
                                bst_p_arr = np.array(bst_p_mid)
                                bst_params.append(bst_p_arr)
                        else:
                            bst_obfn.append(np.full(ts_len, np.nan))
                            if self.best_params is not None:
                                bst_params.append(np.full((len(param_names), ts_len), np.nan))
                    # print(self.dim_names[list(self.dim_names.keys())[0]])  # "field"
                    ds[f"best_obj_function_{ts}"] = (
                        ("objective_functions", self.dim_names[list(self.dim_names.keys())[0]]), np.array(bst_obfn))
                    if self.best_params is not None:
                        ds[f"best_parameter_set_{ts}"] = (
                            ("objective_functions", "parameters", self.dim_names[list(self.dim_names.keys())[0]]),
                            np.array(bst_params))

        if len(self.simulation_results) != 0:

            # reshape the dictionaries.
            sims_list = self.simulation_results
            sims_dict = {}
            for ts in sims_list[0].keys():
                sims_dict[ts] = np.zeros((len(sims_list), len(sims_list[0][ts]), len(sims_list[0][ts][0])))
            for i in range(len(sims_list)):
                for ts, sim_vals in sims_list[i].items():
                    sims_dict[ts][i] = sim_vals

            for ts in objfunc_names.keys():
                ds[f"simulation_results_{ts}"] = (
                    ("repetition", f"observation_{ts}", self.dim_names[list(self.dim_names.keys())[0]]),
                    np.array(sims_dict[ts]))

        for ts in objfunc_names.keys():
            if self.pfactor is not None:
                ds[f"pfactor_{ts}"] = ((self.dim_names[list(self.dim_names.keys())[0]]), self.pfactor[ts])
            if self.rfactor is not None:
                ds[f"rfactor_{ts}"] = ((self.dim_names[list(self.dim_names.keys())[0]]), self.rfactor[ts])
            if self.ppu_lower is not None:
                ds[f"95PPU_lower_{ts}"] = ((f"observation_{ts}", self.dim_names[list(self.dim_names.keys())[0]]),
                                           self.ppu_lower[ts])
            if self.ppu_upper is not None:
                ds[f"95PPU_upper_{ts}"] = ((f"observation_{ts}", self.dim_names[list(self.dim_names.keys())[0]]),
                                           self.ppu_upper[ts])

        coord_dict = {}
        for k, v in ds.sizes.items():
            if k == 'objective_functions':
                coord_dict[k] = (k, objfunc_names_flat)
            elif k == 'parameters':
                coord_dict[k] = (k, param_names)
            else:
                coord_dict[k] = (k, np.arange(v) + 1)

        ds = ds.assign_coords(coord_dict)

        if self.thresholds is not None:
            for k, v in self.thresholds.items():
                if k == 'pfactor_threshold':
                    ds[k] = (('scalar',), np.array([v]))
                elif k == 'min_refined_params_threshold':
                    ds[k] = (('scalar',), np.array([v]))
                else:
                    obfthrs = np.empty(len(objfunc_names_flat))
                    for obj, val in v.items():
                        if obj in objfunc_names_flat:
                            obidx = objfunc_names_flat.index(obj)
                            obfthrs[obidx] = val
                    if not np.isinf(obfthrs).all():
                        ds[f'obj_func_thresholds_{k}'] = (('objective_functions',), obfthrs)

        return ds