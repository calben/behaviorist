import pandas as pd
from behaviorist.util import *


def add_pdfs_to_session(session: dict) -> None:
    session["neuron1pdf"] = session["neuron1"].apply(estimate_density_function_by_window)
    session["neuron2pdf"] = session["neuron2"].apply(estimate_density_function_by_window)


def add_statistical_differences_to_session(session: dict) -> None:
    session["statisticaldifferences"] = session["neuron1pdf"] - session["neuron2pdf"]


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
    experiment["neuron2"] = experiment["neuron2"][:][0:length]


def remove_null_trials_from_session(session: dict) -> None:
    session["statisticaldifferences"] = session["statisticaldifferences"][:,
                                        session["statisticaldifferences"].mean() > 0]


def add_feature_matrix_to_session(session: dict) -> None:
    df = session["statisticaldifferences"].describe().T.drop("count")
    df["skew"] = session["statisticaldifferences"].skew()
    df["label"] = session["sessionleversuccess"][:]
    session["Features"] = df


def add_full_session_values(session: dict) -> None:
    shift_session_by_signal_onset(session)
    add_pdfs_to_session(session)
    add_statistical_differences_to_session(session)
    remove_null_trials_from_session(session)
    add_feature_matrix_to_session(session)
