from behaviorist.io import *
from behaviorist.session import *
from behaviorist.learning import *
import os

all_sessions_features = pd.DataFrame()
all_sessions_params = pd.DataFrame()

for i in range(1, 8):
    session = load_experiment_with_params_to_dataframe("C:/Users/Calem Bendell/Google Drive/Cogs 401/", "%02d" % (i,))
    add_full_session_values(session)
    if(i is 1):
        all_sessions_params = session["params"]
    else:
        all_sessions_params = pd.concat([all_sessions_params, session["params"]])
    if(i is 1):
        all_sessions_features = session["features"]
    else:
        all_sessions_features = pd.concat([all_sessions_features, session["features"]])
    # session_directory = "session-" + "%02d" % (i,) + "/"
    # os.mkdir(session_directory)
    # session["params"].to_csv(session_directory + "params.csv")
    # session["features"].to_csv(session_directory + "features.csv")

all_sessions_params.to_csv("all_session_parameters.csv")
all_sessions_features.to_csv("all_session_features.csv")
all_sessions_features_filtered = all_sessions_features[all_sessions_features["distance-mean"] > 0].dropna()
all_sessions_features_filtered.to_csv("all_session_features_filtered.csv")
all_sessions_features_filtered.corr().to_csv("all_session_features_corr.csv")
all_sessions_features_filtered.describe().to_csv("all_session_features_filtered_description.csv")

test_all_algorithms(load_feature_matrix_for_ml("all_session_features_filtered.csv"))
