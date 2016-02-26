from behaviorist.io import *
from behaviorist.session import *
from behaviorist.learning import *
import os

all_sessions_features = pd.DataFrame()
all_sessions_params = pd.DataFrame()

for i in range(1, 48):
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
    session_directory = "../data/session-" + "%02d" % (i,) + "/"
    os.makedirs(session_directory, exist_ok=True)
    for k, v in session.items():
        v.to_csv(session_directory + k + ".csv")
    session["features"].corr().to_csv(session_directory + "features-corr.csv")
    session["features"].describe().to_csv(session_directory + "features-describe.csv")

if not os.path.exists("../data/"):
    os.mkdir("../data/")

all_sessions_params.to_csv("../data/all_session_parameters.csv")
all_sessions_features.to_csv("../data/all_session_features.csv")
all_sessions_features_filtered = all_sessions_features[all_sessions_features["distance-mean"] > 0].dropna()
all_sessions_features_filtered.to_csv("../data/all_session_features_filtered.csv")
all_sessions_features_filtered.corr().to_csv("../data/all_session_features_corr.csv")
all_sessions_features_filtered.describe().to_csv("../data/all_session_features_filtered_description.csv")
