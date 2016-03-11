import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("ticks")

totals = {}

for i in range(1, 18):
    session_directory = "../data/session-" + "%02d" % (i,) + "/"
    dfs = {}
    for j in range(51):
        dfs[j] = pd.read_csv(session_directory + "corr-signal-time-" + str(j) + "-machinelearningsummary.csv",
                             index_col=0)
    fig, ax = plt.subplots(1, figsize=(5, 5))
    for algo in dfs[0].columns:
        y = [df[algo]["MCC"] for df in dfs.values()]
        if algo not in totals.keys():
            totals[algo] = pd.DataFrame()
        totals[algo]["MCC-" + str(i)] = y
        x = np.linspace(0, 500, len(y))
        ax.plot(x, y, label=algo)
    ax.legend(loc="best")
    sns.despine(offset=5)
    plt.tight_layout()
    plt.savefig(session_directory + "corr-signal-machinelearningsummary.pdf")
    plt.close(fig)

for algo in totals.keys():
    fig, ax = plt.subplots(1, figsize=(5, 5))
    sns.tsplot(data=[totals[algo][col] for col in totals[algo].columns], ax=ax)
    sns.despine(offset=5)
    plt.tight_layout()
    plt.savefig("../data/signal-corr-learning-" + algo + ".pdf")
