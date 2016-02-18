from unittest import TestCase
from behaviorist.learning import *


class TestLearning(TestCase):

    def test_load_feature_matrix_for_ml(self):
        data = load_feature_matrix_for_ml("session/features.csv")
        assert(True)


    def test_run_cross_validation(self):
        print(run_cross_validation(load_feature_matrix_for_ml("session/features.csv")))
        assert(True)


    def test_run_k_cross_validation(self):
        run_k_cross_validation(load_feature_matrix_for_ml("session/features.csv"))
        assert(True)

    def test_test_all_algorithms(self):
        test_all_algorithms(load_feature_matrix_for_ml("session/features.csv"))
        assert(False)