from unittest import TestCase
from behaviorist.io import *
from behaviorist.plotting import *


class TestUtil(TestCase):
    def test_plot_signal_difference(self):
        experiment = load_experiment_with_params_to_dataframe("C:/Users/Calem Bendell/Google Drive/Cogs 401/", "01")
        a = experiment["neuron1"][0][750:850]
        b = experiment["neuron2"][0][750:850]
        plot_signal_difference(a, b, "plot_signal_difference_test.pdf", bandwidth=10)

    def test_plot_signal_windows_for_train(self):
        experiment = load_experiment_with_params_to_dataframe("C:/Users/Calem Bendell/Google Drive/Cogs 401/", "01")
        data = experiment["neuron1"][0][600:900]
        plot_signal_windows_for_train(data, "plot_signal_windows_for_train.pdf")
