from behaviorist.learning import *

for i in range(2, 10):
    test_all_algorithms("Session-" + "%02d" % (i,),
                        load_feature_matrix_for_ml("../data/session-" + "%02d" % (i,) + "/features.csv"),
                        output="../data/session-" + "%02d" % (i,) + "/machinelearningsummary")
