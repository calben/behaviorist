import pandas as pd
import numpy as np
from scipy import signal


def estimate_density_function_by_window(a: pd.Series, window=None, bandwidth=None) -> pd.Series:
    if window is None:
        window = signal.parzen
    if bandwidth is None:
        bandwidth = 30
    window = window(bandwidth)
    estimate = signal.convolve(a, window, mode="same")
    return estimate


def get_session_lever_delay(params: pd.DataFrame) -> pd.Series:
    return params["LeverUp"] - params["SignalOn"]


def get_session_lever_success(lever_delay: pd.Series, time_min=200, time_max=800) -> pd.Series:
    lever_success = lever_delay[:].copy()
    lever_success[:] = 1
    lever_success[lever_delay < time_min] = 0
    lever_success[lever_delay > time_max] = 0
    return lever_success
