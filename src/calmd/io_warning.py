"""

"""
import warnings
import inspect
import os

def user_warning(msg: str, frame: inspect.getframeinfo, wtype: tuple = UserWarning) -> None:
    """
    Method to standardize the warning output and avoid
    absolute file paths in warning messages - copied from pygsflow repository

    Parameters
    ----------
    msg : str
        error message
    frame : named tuple
        from inspect.getframeinfo
    wtype :
        warning type to be displayed defaults to UserWarning
    """
    module = os.path.split(frame.filename)[-1]
    warnings.warn_explicit(msg, wtype, module, frame.lineno)