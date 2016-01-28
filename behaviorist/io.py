import pandas as pd
from matlabconverters.loaders import strip_mat_metadata, load_mat
import os

def load_experiment(experiment_path : str) -> {}:
    experiment = {}
    for f in os.listdir(experiment_path):
        experiment[f[:-3]] = pd.read_csv(experiment_path + f, index_col = 0)
    return experiment


def preprocess_directory_of_raw_mats(directory : str) -> None:
    for f in os.listdir(directory):
        if f[-3:] == "mat":
            if "Params" in f:
                out_prefix = f[:-4] + "/raw/"
                out_prefix = out_prefix.replace("Params", "")
            else:
                out_prefix = f[:-4] + "/raw/"
            data = strip_mat_metadata(load_mat(f, False))
            if not os.path.exists(out_prefix):
                os.makedirs(out_prefix)
            for k, v in data.items():
                df = pd.DataFrame(v)
                if "neuron" in k:
                    df.transpose()
                df.to_csv(out_prefix + k + ".csv")