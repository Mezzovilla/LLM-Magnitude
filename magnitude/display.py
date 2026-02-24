import pandas as pd
import numpy as np

def display_pd(
        df: pd.DataFrame,
        digits: int = 2,
        zero_char: str = ".",
        ncol: int = 3,
    ) -> pd.DataFrame:
    """
    Pretty display of a square pandas DataFrame.
    """
    if df.shape[0] != df.shape[1]:
        raise ValueError("display_pd expects a square DataFrame")

    n = df.shape[1]
    df_round = df.round(digits)

    # Case: small enough, show everything
    if n <= 2 * ncol:
        df_show = df_round.copy()
    else:
        left = df_round.iloc[:, :ncol]
        right = df_round.iloc[:, -ncol:]

        # Create ellipsis column
        ellipsis = pd.DataFrame(
            [["…"]] * df.shape[0],
            index=df.index,
            columns=["…"]
        )

        df_show = pd.concat([left, ellipsis, right], axis=1)

    # Format values
    def format_value(x):
        if isinstance(x, str):
            return x
        if np.isclose(x, 0.0):
            return zero_char
        return f"{x:.{digits}f}"

    df_fmt = df_show.map(format_value)

    return df_fmt

import sys

def progress_bar(current: int, total: int, bar_width: int = 40):
    if total <= 0:
        raise ValueError("total must be a positive integer")

    # Clamp current to [0, total]
    current = max(0, min(current, total))

    progress = current / total
    filled = int(bar_width * progress)
    bar = "█" * filled + "-" * (bar_width - filled)
    percent = progress * 100

    sys.stdout.write(f"\r[{bar}] {percent:6.2f}% ({current}/{total})")
    sys.stdout.flush()

    if current == total:
        print()  # newline at completion
