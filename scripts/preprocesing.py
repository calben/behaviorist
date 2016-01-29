import os

import numpy as np
import pandas as pd

def convert_session_to_kde_files():
    load_mat_to_pandas()

if __name__ == '__main__':
    directory = "C:/Users/Calem Bendell/Google Drive/Cogs 401/"
    for i in range(1, 4):
            data = strip_mat_metadata(load_mat(f, False))
            if not os.path.exists(out_prefix):
                os.makedirs(out_prefix)
            for k, v in data.items():
                df = pd.DataFrame(v)
                if "neuron" in k:
                    df.transpose()
                df.to_csv(out_prefix + k + ".csv")    p = Process(target=convert_session_to_kde_files, args=(i, targets[i], 300))
    p.start()
    print("Started P", i, "for", targets[i])