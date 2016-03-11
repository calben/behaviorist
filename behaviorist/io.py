import os

import pandas as pd
from matlabconverters.loaders import strip_mat_metadata, load_mat

from behaviorist.util import get_session_lever_delay, get_session_lever_success


def load_session(directory: str, session_name: str) -> pd.DataFrame:
    session = {}
    experiment_fname = directory + "session" + session_name + ".mat"
    params_fname = directory + "sessionParams" + session_name + ".mat"
    experiment_dict = strip_mat_metadata(load_mat(experiment_fname, False))
    for k, v in experiment_dict.items():
        if "neuron" in k:
            df = pd.DataFrame(v).transpose()
            session[k] = df
    params_data = strip_mat_metadata(load_mat(params_fname, False))["params"]
    session["params"] = pd.DataFrame(params_data, columns=["StimOn", "SignalOn", "LeverUp", "Coh1", "Coh2"]).dropna()
    session["params"]["LeverDelay"] = get_session_lever_delay(session["params"])
    session["params"]["LeverSuccess"] = get_session_lever_success(session["params"]["LeverDelay"])
    session["params"]["Session"] = pd.Series([session_name] * len(session["params"]))
    session["params"].index.name = "Trial"
    return session


def load_session_corr_signal(directory: str, session_name: str) -> pd.DataFrame:
    scs = {}
    session_fname = directory + "sessionCorrSignal" + session_name + ".mat"
    session_dict = strip_mat_metadata(load_mat(session_fname))
    for k, v in session_dict.items():
        scs[k] = pd.DataFrame(v)
    for col in scs["corrC"].columns:
        scs["time-" + str(col)] = pd.DataFrame(
                {"correlation": scs["corrC"][col], "mutual_information": scs["MIC"][col],
                 "label": [1] * len(scs["corrC"][col])})
        scs["time-" + str(col)] = scs["time-" + str(col)].append(
                pd.DataFrame({"correlation": scs["corrF"][col], "mutual_information": scs["MIF"][col],
                              "label": [0] * len(scs["corrF"][col])})).reset_index(drop=True)
    return scs


def preprocess_directory_of_raw_mats(directory: str) -> None:
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


def load_feature_matrix_for_ml(csv: str, balance_input=True, balance_by_sample=True) -> dict:
    df = pd.read_csv(csv, index_col=0)
    df = filter_feature_matrix(df, balance_input=balance_input, balance_by_sample=balance_by_sample)
    data = {}
    data["target"] = df["label"].values
    data["features"] = df.drop("label", axis=1).values
    return data


def filter_feature_matrix(df: pd.DataFrame, balance_input=True, balance_by_sample=True) -> pd.DataFrame:
    df = df.dropna(axis=0)
    # df = df[df["distance-mean"] != 0]
    if balance_input:
        mat_pos = df[df["label"] == 1]
        mat_neg = df[df["label"] == 0]
        print(len(mat_pos), "positive samples")
        print(len(mat_neg), "negative samples")
        if balance_by_sample:
            if len(mat_pos) >= len(mat_neg):
                mat_pos = mat_pos.sample(len(mat_neg))
            else:
                mat_neg = mat_neg.sample(len(mat_pos))
        else:
            if len(mat_pos) >= len(mat_neg):
                mat_pos = mat_pos[:len(mat_neg)]
            else:
                mat_neg = mat_neg[:len(mat_pos)]

    df = pd.concat([mat_pos, mat_neg])
    df = df.reset_index(drop=True)
    # df = df[["label", "distance-std", "distance-25%", "distance-75%", "correlation-abs"]]
    return df
