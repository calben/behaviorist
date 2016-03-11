import os

from behaviorist.io import *
from behaviorist.learning import *

all_sessions_features = pd.DataFrame()
all_sessions_params = pd.DataFrame()

for i in range(1, 48):
    session = load_session_corr_signal("C:/Users/Calem Bendell/Google Drive/Cogs 401/", str(i))
    session_directory = "../data/session-" + "%02d" % (i,) + "/"
    os.makedirs(session_directory, exist_ok=True)
    for k, v in session.items():
        v.to_csv(session_directory + "corr-signal-" + k + ".csv", float_format="%.4f")
