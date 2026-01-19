import pandas as pd
import numpy as np
from rich import print
import typing as t
import time

from magnitude.inverses import (
    MatrixInverter,
    INVERSION_METHODS
)

def magnitude_of_matrix(
        P: np.ndarray, 
        method: str = "direct", 
        check_invertibility: bool = False, 
        methods: t.Dict[str, MatrixInverter] = INVERSION_METHODS, 
        verbose: bool = False,
        digits: int = 3
    ) -> float:
    """
    Calcula a magnitude da matriz P, i.e., soma dos elementos de P^{-1}.
    """
    if method not in methods.keys():
        raise ValueError(f"Method '{method}' not recognized. Available methods: {list(methods.keys())}")

    if check_invertibility:
        rank = np.linalg.matrix_rank(P)
        if rank < P.shape[0]:
            print("Matriz P:")
            print(f"{P.shape=}")
            print(pd.DataFrame(P))
            raise ValueError(f"Matrix P is not invertible. Rank deficient: {rank=} < {P.shape[0]=}")
    
    P_inv = methods[method](P)

    if verbose:
        display = lambda x: pd.DataFrame(x).round(digits).replace(0.0, '.').to_string()
        print("Matriz P:")
        print(display(P))
        print("> Inverse matrix")
        print(display(P_inv))

    return float(P_inv.sum())

def compare_magnitude_methods(P: np.ndarray):
    mag_geo = magnitude_of_matrix(P, method="geometric")
    mag_dir = magnitude_of_matrix(P, method="direct")

    print(f"Magnitude (geometric): {mag_geo}")
    print(f"Magnitude (direct): {mag_dir}")

    assert np.isclose(mag_geo, mag_dir), "Magnitudes from different methods do not match!"

def benchmark_magnitude_methods(
        P: t.Union[pd.DataFrame, np.ndarray], 
        repeat: int = 5, 
        methods: t.Dict[str, MatrixInverter] = INVERSION_METHODS,
        verbose: bool = True
    ) -> pd.DataFrame:
    times = {
        method: [0.0 for _ in range(repeat)] for method in methods.keys()
    }

    if isinstance(P, pd.DataFrame):
        P = P.values
    assert isinstance(P, np.ndarray)

    for i in range(repeat):
        for method in times.keys():
            start = time.perf_counter()
            _ = magnitude_of_matrix(P, method=method)
            times[method][i] = time.perf_counter() - start

    # start empty dataframe for mean/deviation for each method
    df = pd.DataFrame({
        "Method": methods.keys(),
        "Mean": [np.mean(times[method]) for method in times.keys()],
        "Std Dev": [np.std(times[method]) for method in times.keys()]
    }).set_index("Method")

    if not verbose:
        return df

    print(f"Tempo médio ({repeat=}) por método:")
    for method in times.keys():
        deviation = df.at[method, "Std Dev"]
        mean = df.at[method, "Mean"]

        print(f"-  {(method+':').ljust(15)} {mean:.6f} s ± {deviation:.6f} s")

    return df


if __name__ == "__main__":
    N = 1550

    T = np.triu(np.random.rand(N, N))
    np.fill_diagonal(T, 1.0)
    
    benchmark_magnitude_methods(T)