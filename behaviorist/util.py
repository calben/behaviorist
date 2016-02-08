import pandas as pd
import numpy as np
from scipy import signal


def get_statistical_difference(a: pd.Series, b: pd.Series) -> pd.Series:
    difference = pd.Series(a - b).apply(np.abs)
    return difference


def estimate_density_function_by_window(a: pd.Series, window=None, bandwidth=None) -> pd.Series:
    if window is None:
        window = signal.parzen
    if bandwidth is None:
        bandwidth = 30
    window = window(bandwidth)
    estimate = signal.convolve(a, window, mode="same")
    return estimate


def get_session_lever_delay(params: pd.DataFrame) -> pd.Series:
    return params["LeverUp"] - params["StimOn"]


def get_session_lever_success(lever_delay: pd.Series, time=500) -> pd.Series:
    lever_delay[lever_delay < 0] = 0
    lever_delay[lever_delay > time] = 0
    lever_delay[lever_delay > 0] = 1
    return lever_delay


def shift_session_by_signal_onset(experiment: dict, length=500) -> dict:
    for i in range(len(experiment["params"]["SignalOn"])):
        offset = experiment["params"]["SignalOn"][i]
        if offset < length:
            experiment["neuron1"][i][:] = 0
            experiment["neuron2"][i][:] = 0
        else:
            experiment["neuron1"][i].shift(-(offset - length))
            experiment["neuron2"][i].shift(-(offset - length))
    experiment["neuron1"] = experiment["neuron1"][:][0:length]
    experiment["neuron2"] = experiment["neuron1"][:][0:length]
