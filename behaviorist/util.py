import pandas as pd
import numpy as np
from scipy import signal


def get_statistical_difference(a:pd.Series, b:pd.Series) -> pd.Series:
    difference = pd.Series(a - b).apply(np.abs)
    return difference


def estimate_density_function_by_window(a : pd.Series, window=None, bandwidth=None) -> pd.Series:
    if window is None:
        window = signal.parzen
    if bandwidth is None:
        bandwidth = 30
    window = window(bandwidth)
    estimate = signal.convolve(a, window, mode="same")
    return estimate


def get_session_success(params:pd.DataFrame, time=250):
    success = params["StimOn"]
