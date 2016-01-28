import pandas as pd
import numpy as np
from scipy import stats

def convert_neuron_discrete_to_gaussian_estimation(signal : pd.DataFrame) -> pd.DataFrame:
    kernels = {}
    for k, v in signal.items():
