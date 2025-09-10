import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


def plot_training_metrics(save_dir: str, metrics: list[str]) -> plt.Figure:
    """Plot training metrics from latest lightning CSVLogger version.

    Args:
        save_dir: path to save directory of metrics.csv
        metrics: list of metrics to plot
    """
    metrics_path = os.path.join(save_dir, "lightning_logs", "metrics.csv")

    df = pd.read_csv(metrics_path)

    plot_metric = {}
    for m in metrics:
        try:
            plot_metric[m] = df[df[m].notna()][m]
        except KeyError:
            print(f"{m} not in metrics, available are {df.columns}")

    fig, ax = plt.subplots(ncols=len(metrics), figsize=(5 * len(metrics), 5))
    ax = np.atleast_1d(ax)
    for idx, (m, p) in enumerate(plot_metric.items()):
        ax[idx].plot(p)
        ax[idx].set_title(m)
    return fig