from unittest import TestCase
from behaviorist.io import *
from behaviorist.util import *

class TestUtil(TestCase):

    def test_get_statistical_difference(self):
        experiment = load_experiment_with_params_to_dataframe("C:/Users/Calem Bendell/Google Drive/Cogs 401/", "01")
        pdf = estimate_density_function_by_window(experiment["neuron1"][0][0:600])
        print(pdf)
        assert(True)

    def test_get_statistical_difference(sefl):
        experiment = load_experiment_with_params_to_dataframe("C:/Users/Calem Bendell/Google Drive/Cogs 401/", "01")
        a = experiment["neuron1"][0][0:600]
        b = experiment["neuron2"][0][0:600]
        a_pdf = estimate_density_function_by_window(a)
        b_pdf = estimate_density_function_by_window(b)
        diff = get_statistical_difference(a_pdf, b_pdf)
        print(diff)
        assert(False)