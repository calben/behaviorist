import pandas as pd
from behaviorist.util import estimate_density_function_by_window


def add_pdfs_to_session(session: dict):
    session["neuron1pdf"] = session["neuron1"].apply(estimate_density_function_by_window)
    session["neuron2pdf"] = session["neuron2"].apply(estimate_density_function_by_window)
