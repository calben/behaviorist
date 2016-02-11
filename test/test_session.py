from unittest import TestCase
from behaviorist.io import *
from behaviorist.session import *
from behaviorist.plotting import *

class TestSession(TestCase):

    def test_add_feature_matrix_to_session(self):
        experiment = load_experiment_with_params_to_dataframe("C:/Users/Calem Bendell/Google Drive/Cogs 401/", "01")
        shift_session_by_signal_onset(experiment)
        assert(True)


    def test_shift_session_by_signal_onset(self):
        experiment = load_experiment_with_params_to_dataframe("C:/Users/Calem Bendell/Google Drive/Cogs 401/", "01")
        shift_session_by_signal_onset(experiment)
        assert(True)

    def test_get_session_lever_success(self):
        times = pd.Series(np.asarray([4040, 45, 433]))
        success = get_session_lever_success(times, time_min=200, time_max=800)
        print(times)
        print(success)
        assert((times.values == pd.Series(np.asarray([4040, 45, 433])).values).all())
        assert((success.values == pd.Series(np.asarray([0, 0, 1])).values).all())


    def test_add_full_session_values(self):
        assert(True)
