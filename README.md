# calmd
***A multi-dimensional and model-independent calibration package***

calmd uses MS-CUA (multicriteria sequential calibration and uncertainty analysis). For more information, see [Wu et al. (2021)](https://www.nature.com/articles/s41598-021-96250-6) or the included example notebooks.

### Environment

This package will likely be used in the same environment as your model, and as such, is designed to be lightweight. Dependencies are:
- dask
- matplotlib
- numpy
- scipy
- [spotpy](https://spotpy.readthedocs.io/en/latest/)
- tqdm
- xarray
- ipykernel (to run example notebooks)