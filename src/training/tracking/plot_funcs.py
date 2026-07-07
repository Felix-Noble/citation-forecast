import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def histplot(prefix: str, x: np.ndarray):
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(9, 5))

    sns.histplot(
        x,
        kde=True,
        color="#1f77b4",
        edgecolor="white",
        linewidth=1.2,
        alpha=0.6,
        bins="auto",
    )

    plt.title(
        f"Distribution of Values ({prefix})",
        fontsize=14,
        pad=15,
    )
    plt.xlabel("Value", fontsize=12, labelpad=10)
    plt.ylabel("Count / Density", fontsize=12, labelpad=10)
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    return fig
