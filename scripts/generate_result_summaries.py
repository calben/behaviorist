import sys
import os
import pandas as pd


all_sessions_result = pd.DataFrame()

for i in range(1, 48):
    result = pd.read_csv("../data/session-" + "%02d" % (i,) + "/machinelearningsummary.csv", index_col=0)
    result = result.T
    if(i is 1):
        all_sessions_result = result
    else:
        all_sessions_result = pd.concat([all_sessions_result, result])
    session_directory = "../data/session-" + "%02d" % (i,) + "/"
    os.makedirs(session_directory, exist_ok=True)

all_sessions_result.describe().to_csv("../data/all-session-results-summary.csv")
