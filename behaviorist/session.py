import pandas as pd
import numpy as np
from behaviorist.util import *


def add_pdfs_to_session(session: dict) -> None:
    session["neuron1pdf"] = session["neuron1"].apply(estimate_density_function_by_window)
    session["neuron2pdf"] = session["neuron2"].apply(estimate_density_function_by_window)


def add_statistical_differences_to_session(session: dict) -> None:
    session["statisticaldifferences"] = session["neuron1pdf"] - session["neuron2pdf"]


def shift_session_by_signal_onset(experiment: dict, length=300) -> dict:
    for i in range(len(experiment["params"]["SignalOn"])):
        offset = experiment["params"]["SignalOn"][i]
        if offset < length:
            experiment["neuron1"][i][:] = 0
            experiment["neuron2"][i][:] = 0
        else:
            experiment["neuron1"][i].shift(-(offset - length))
            experiment["neuron2"][i].shift(-(offset - length))
    experiment["neuron1"] = experiment["neuron1"][:][0:length]
    experiment["neuron2"] = experiment["neuron2"][:][0:length]


def remove_null_trials_from_session(session: dict) -> None:
    diff = session["statisticaldifferences"]
    session["statisticaldifferences"] = diff.ix[:, diff.mean() > 0]


def add_feature_matrix_to_session(session: dict) -> None:
    df = session["statisticaldifferences"].describe().T
    df.index.name = "Trial"
    df = df.drop("count", axis=1)
    df.columns = ["distance-" + x for x in df.columns]
    df["Session"] = pd.Series([session["params"]["Session"][0]]*len(df))
    df["distance-skew"] = session["statisticaldifferences"].skew()
    df["correlation"] = get_pairwise_corr_between_two_dataframes(session["neuron1pdf"], session["neuron2pdf"])
    df["label"] = session["leversuccess"][:]
    session["features"] = df


def get_pairwise_corr_between_two_dataframes(a, b):
    corrcoefs = []
    for col in a.columns:
        corrcoefs.append(np.abs(np.corrcoef(a[col], b[col])[0][1]))
    return pd.Series(corrcoefs)


def add_lever_values_to_session(session: dict) -> None:
    session["leverdelay"] = get_session_lever_delay(session["params"])
    session["leversuccess"] = get_session_lever_success(session["leverdelay"])


def add_full_session_values(session: dict) -> None:
    shift_session_by_signal_onset(session)
    add_pdfs_to_session(session)
    add_statistical_differences_to_session(session)
    add_lever_values_to_session(session)
    add_feature_matrix_to_session(session)
