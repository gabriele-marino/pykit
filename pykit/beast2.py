import os
from pathlib import Path
from typing import Callable

import arviz as az
import numpy as np
import pandas as pd
from tqdm import tqdm


def process_log_df(df: pd.DataFrame, hdi_prob: float = 0.95) -> dict[str, float]:
    log_summary = {}
    for column in df.columns:
        log_summary[f"{column}_mean"] = df[column].mean()
        log_summary[f"{column}_median"] = df[column].median()
        log_summary[f"{column}_min"] = df[column].min()
        log_summary[f"{column}_max"] = df[column].max()
        log_summary[f"{column}_std"] = df[column].std()
        log_summary[f"{column}_ess"] = az.ess(np.array(df[column]))
        lower, upper = az.hdi(np.array(df[column]), hdi_prob=hdi_prob)
        log_summary[f"{column}_lower"] = lower
        log_summary[f"{column}_upper"] = upper
    log_summary["n_samples"] = len(df)
    return log_summary


def read_log(
    log_file: str,
    burn_in: float = 0.1,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(log_file, sep="\t", comment="#")
    df = df[df["Sample"] >= burn_in * len(df)]
    df = df.drop(columns=["Sample"])
    if columns is not None:
        df = df[columns]
    return df


def read_logs(
    logs_dir: str,
    burn_in: float = 0.1,
    hdi_prob: float = 0.95,
    columns: list[str] | None = None,
    preprocessing_func: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if preprocessing_func is None:
        preprocessing_func = lambda x: x
    return pd.DataFrame(
        [
            process_log_df(
                df=preprocessing_func(
                    read_log(
                        log_file=os.path.join(logs_dir, log_file),
                        burn_in=burn_in,
                        columns=columns,
                    )
                ),
                hdi_prob=hdi_prob,
            )
            | {"id": Path(log_file).stem}
            for log_file in tqdm(os.listdir(logs_dir))
            if log_file.endswith(".log")
        ]
    ).set_index("id")
