import pandas as pd
import numpy as np
from behaviorist.util import *


def add_pdfs_to_session(session: dict) -> None:
    session["neuron1pdf"] = session["neuron1"].apply(estimate_density_function_by_window)
    session["neuron2pdf"] = session["neuron2"].apply(estimate_density_function_by_window)


def add_statistical_differences_to_session(session: dict) -> None:
    df = session["neuron1pdf"] - session["neuron2pdf"]
    df = df.abs()
    df = df / df.mean()
    session["statisticaldifferences"] = df


def remove_session_resuls_without_parameters(session: dict) -> None:
    cols_to_trim = ["neuron1", "neuron2"]
    for col in cols_to_trim:
        session[col] = session[col].ix[:, :(len(session["params"]) - 1)]


def shift_session_by_signal_onset(experiment: dict, length=300) -> dict:
    cols_to_shift = ["neuron1", "neuron2", "neuron1pdf", "neuron2pdf"]
    for i in range(len(experiment["params"]["SignalOn"])):
        offset = experiment["params"]["SignalOn"][i].astype(np.int)
        for col in cols_to_shift:
            if offset < length:
                experiment[col].drop(i, axis=1, inplace=True)
            else:
                experiment[col][i] = experiment[col][i][offset - 300:offset].reset_index(drop=True)
    for col in cols_to_shift:
        experiment[col] = experiment[col][:][:length]


def remove_null_trials_from_session(session: dict) -> None:
    diff = session["statisticaldifferences"]
    session["statisticaldifferences"] = diff.ix[:, diff.mean() > 0]


def add_feature_matrix_to_session(session: dict) -> None:
    df = session["statisticaldifferences"].describe().T
    print(df)
    df.index.name = "Trial"
    df = df.drop("count", axis=1)
    df.columns = ["distance-" + x for x in df.columns]
    df["session"] = session["params"]["Session"]
    df["distance-skew"] = session["statisticaldifferences"].skew()
    df["correlation"] = get_pairwise_corr_between_two_dataframes(session["neuron1pdf"], session["neuron2pdf"])
    df["correlation-abs"] = df["correlation"].abs()
    df["label"] = session["params"]["LeverSuccess"][:]
    session["features"] = df


def get_pairwise_corr_between_two_dataframes(a, b):
    corrcoefs = {}
    for col in a.columns:
        corrcoefs[col] = np.corrcoef(a[col], b[col])[0][1]
    return pd.Series(corrcoefs)


def add_lever_values_to_session(session: dict) -> None:
    session["params"]["LeverDelay"] = get_session_lever_delay(session["params"])
    session["params"]["LeverSuccess"] = get_session_lever_success(session["params"]["LeverDelay"])


def add_full_session_values(session: dict) -> None:
    remove_session_resuls_without_parameters(session)
    add_pdfs_to_session(session)
    shift_session_by_signal_onset(session)
    add_statistical_differences_to_session(session)
    add_lever_values_to_session(session)
    add_feature_matrix_to_session(session)
