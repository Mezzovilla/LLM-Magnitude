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