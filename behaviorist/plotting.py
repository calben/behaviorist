import matplotlib.pyplot as plt
from behaviorist.util import *
from scipy import signal
import seaborn as sns

sns.set_style("ticks")


def plot_signal_windows_for_train(sig: pd.DataFrame, plot_name: str) -> None:
    sig = sig.reset_index(drop=True)

    par_win10 = signal.parzen(10)
    par_win30 = signal.parzen(30)
    han_win10 = signal.hanning(10)
    han_win30 = signal.hanning(30)

    par_filtered10 = signal.convolve(sig, par_win10, mode="same")
    par_filtered30 = signal.convolve(sig, par_win30, mode="same")
    han_filtered10 = signal.convolve(sig, han_win10, mode="same")
    han_filtered30 = signal.convolve(sig, han_win30, mode="same")

    fig, ax = plt.subplots(1)

    ax.plot(par_filtered10, label="Parzen 10")
    ax.plot(par_filtered30, label="Parzen 30")
    ax.plot(han_filtered10, label="Han 10")
    ax.plot(han_filtered30, label="Han 30")

    ax.set_title("Density Estimation by Multiple Windows")
    ax.legend()

    ax.plot(sig[sig > 0] + 4, "|", label="Signal", ms=50, color="black")

    sns.despine(offset=5)

    plt.savefig(plot_name)
    plt.close(fig)


def plot_signal_difference(a: pd.Series, b: pd.Series, plot_name: str, get_density=True, window=None, bandwidth=None):
    if get_density:
        a = estimate_density_function_by_window(a, window=window, bandwidth=bandwidth)
        b = estimate_density_function_by_window(b, window=window, bandwidth=bandwidth)

    diff = get_statistical_difference(a, b)
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True, figsize=(10, 8))
    ax1.plot(a, label="A", color="blue")
    ax2.plot(b, label="B", color="purple")
    ax3.plot(a, label="A", color="blue", alpha=0.2)
    ax3.plot(b, label="B", color="purple", alpha=0.2)
    ax3.plot(diff, label="DIFF", color="red")
    ax3.legend()

    ax1.set_title("Signal A")
    ax2.set_title("Signal B")
    ax3.set_title("Statistical Difference")

    sns.despine(offset=5)
    plt.tight_layout()

    plt.savefig(plot_name)
    plt.close(fig)


def plot_experiment_to_axis(experiment: dict, ax: plt.Axes) -> None:
    experiment["neuron1"].plot(kind="kde")
