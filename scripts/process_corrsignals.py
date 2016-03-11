from behaviorist.io import load_feature_matrix_for_ml
from behaviorist.learning import *

for i in range(1, 48):
    for j in range(51):
        data = load_feature_matrix_for_ml("../data/session-" + "%02d" % (i,) + "/corr-signal-time-" + str(j) + ".csv",
                                          balance_by_sample=False)
        test_all_algorithms("session" + "%02d" % (i,) + "corrsignal" + "%02d" % (j,), data,
                            output="../data/session-" + "%02d" % (i,) + "/corr-signal-time-" + str(
                                    j) + "-machinelearningsummary")
