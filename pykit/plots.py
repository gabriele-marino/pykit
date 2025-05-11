from typing import Any, Callable

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


def plot_metric_by_bin(
    ax: Axes,
    df: pd.DataFrame,
    binning_var: str,
    n_bins: int,
    bin_aggregation_fun: Callable[[pd.DataFrame], float],
    plot_kwargs: dict[str, Any] | None = None,
    binning_var_name: str = "Value",
    metric_name: str = "Metric",
) -> None:
    df = df.copy()
    df["bins"] = pd.qcut(df[binning_var], q=n_bins)
    metric_by_bin = df.groupby("bins", observed=False).apply(bin_aggregation_fun)
    bin_midpoints = df.groupby("bins", observed=False)[binning_var].median()
    ax.plot(
        bin_midpoints,
        metric_by_bin,
        **plot_kwargs if plot_kwargs is not None else {},
    )

    ax.set_xlabel(f"{binning_var_name} (binned midpoint)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} by {binning_var_name}")


def plot_agreement(
    ax: Axes,
    true_values: list[float],
    pred_values: list[float],
    lower_CIs: list[float] | None = None,
    upper_CIs: list[float] | None = None,
    var_name: str = "Value",
) -> None:
    for i, (true_value, pred_value) in enumerate(zip(true_values, pred_values)):
        if lower_CIs is None:
            if upper_CIs is None:
                color = "black"
            else:
                color = "blue" if true_value <= upper_CIs[i] else "red"
                ax.vlines(true_value, pred_value, upper_CIs[i], color=color, alpha=0.2)
        else:
            lower_CI = lower_CIs[i]
            if upper_CIs is None:
                color = "blue" if lower_CI <= true_value else "red"
                ax.vlines(true_value, lower_CI, pred_value, color=color, alpha=0.2)
            else:
                color = "blue" if lower_CI <= true_value <= upper_CIs[i] else "red"
                ax.vlines(true_value, lower_CI, upper_CIs[i], color=color, alpha=0.2)
        ax.scatter(true_value, pred_value, color=color)

    min_val = min(min(true_values), min(pred_values))
    max_val = max(max(true_values), max(pred_values))
    ax.plot(
        [min_val, max_val], [min_val, max_val], color="black", label="Identity line"
    )

    slope, intercept = np.polyfit(true_values, pred_values, deg=1)
    reg_x = np.array([min_val, max_val])
    reg_y = slope * reg_x + intercept
    ax.plot(reg_x, reg_y, "k--", label="Regression line")

    ax.set_xlabel(f"True {var_name}")
    ax.set_ylabel(f"Predicted {var_name}")
    ax.set_title(f"{var_name} agreement")
    ax.legend()


def build_report(figs: list[Figure], output_filepath: str) -> None:
    with PdfPages(output_filepath) as pdf:
        for fig in figs:
            pdf.savefig(fig)
