import pandas as pd
import numpy as np
from rich import print
from dataclasses import dataclass
import typing as t
import time

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from magnitude.graphs import plot_graph
from magnitude.language import (
    is_extension, 
    generate_all_sentences,
    random_language,
    ALPHABET
)
from magnitude.probability import (
    build_P,
    to_pandas,
    display_pd,
    MASS_FUNCTIONS,
    MassFunction
)
from magnitude.inverses import (
    MatrixInverter,
    INVERSION_METHODS
)
from magnitude.invariant import (
    magnitude_of_matrix,
    benchmark_magnitude_methods
)

# ==========================================

@dataclass
class MetricSpace:
    labels: t.Optional[t.List[str]] = None
    similarity: t.Optional[t.Union[pd.DataFrame, np.ndarray]] = None
    scale: float = 1.0
    inv_method: t.Optional[str] = None
    mass_func: t.Callable = MASS_FUNCTIONS["freq_probabilities"]

    def __post_init__(self):
        if self.labels is None and self.similarity is None:
            raise ValueError("User must provid labels or similarity matrix")
        
        if self.labels is None and self.similarity is not None and isinstance(self.similarity, pd.DataFrame):
            self.labels = list(map(str, self.similarity.columns))

        if self.similarity is not None and isinstance(self.similarity, pd.DataFrame):
            self.similarity = self.similarity.values
        
        if self.labels is None:
            assert self.similarity is not None
            self.labels = [f'X{i}' for i in range(self.similarity.shape[0])]
        
        assert isinstance(self.labels, list)

        if self.similarity is None:
            self.similarity = build_P(self.labels, self.scale, mass_func=self.mass_func)
        assert self.similarity is not None
        assert isinstance(self.similarity, np.ndarray)

        # detect if similarity is triangular upper
        is_triangular = np.allclose(self.similarity, np.triu(self.similarity))
        if is_triangular and (self.inv_method is None):
            self.inv_method = "scipy-tri-inv"
        if (not is_triangular) and (self.inv_method is None):
            self.inv_method = "scipy-pinv"

        self.points: int = self.similarity.shape[0]
    
    def get_magnitude(self, inv_method: t.Optional[str] = None, verbose: bool = False) -> float:
        if inv_method is None:
            inv_method = self.inv_method
        assert inv_method is not None

        assert self.similarity is not None
        if isinstance(self.similarity, pd.DataFrame):
            self.similarity = self.similarity.values
        
        
        return magnitude_of_matrix(self.similarity, method=inv_method, verbose=verbose)
    
    def get_magnitude_mc(
            self,
            n_samples: int = 10_000,
            seed: t.Optional[int] = None,
        ) -> float:
        """
        Estima a magnitude via Monte Carlo usando o estimador probabilístico
        baseado na paridade do comprimento do passeio aleatório.

        Parâmetros
        ----------
        n_samples : int
            Número de amostras de Monte Carlo.
        seed : int, opcional
            Semente do gerador aleatório.

        Retorna
        -------
        float
            Estimativa da magnitude.
        """
        assert self.similarity is not None
        if isinstance(self.similarity, pd.DataFrame):
            self.similarity = self.similarity.values

        rng = np.random.default_rng(seed)
        P = self.similarity
        n = self.points

        even_count = 0

        for _ in range(n_samples):
            # estado inicial uniforme
            i = rng.integers(0, n)
            length = 0

            while True:
                # probabilidades da linha i (somente j > i)
                probs = P[i, i + 1 :]
                if probs.size == 0:
                    break

                # se a linha não tem massa (caso degenerado)
                s = probs.sum()
                if s <= 0.0:
                    break

                # amostra do próximo estado
                j_offset = rng.choice(len(probs), p=probs / s)
                i = i + 1 + j_offset
                length += 1

            if length % 2 == 0:
                even_count += 1

        return n * (even_count / n_samples)
    
    def __repr__(self, digits: int = 3):
        assert self.similarity is not None
        if isinstance(self.similarity, pd.DataFrame):
            self.similarity = self.similarity.values

        N: int = self.similarity.shape[0]
        mag = self.get_magnitude()

        matrix: str = display_pd(
            to_pandas(self.similarity, self.labels), 
            digits=digits
        ).to_string()
        return f"MetricSpace(magnitude={mag:.3f}, points={self.points}):\n{matrix}"
    
    def similarity_pd(self) -> pd.DataFrame:
        assert self.similarity is not None
        if isinstance(self.similarity, pd.DataFrame):
            self.similarity = self.similarity.values
        return to_pandas(self.similarity, self.labels)

    def display(self, **kargs) -> None:
        print(self.__repr__(**kargs))

    # |tA|, t > 1
    # ===============
    def magnitude_curve(self, scales: t.Union[t.List[float], np.ndarray] = np.linspace(0.01, 10.0, 50)) -> pd.DataFrame:
        # empty dataframe
        df = pd.DataFrame({
            "Scale": scales,
            "Magnitude": [0.0 for _ in scales]
        })
        assert self.inv_method is not None
        assert self.similarity is not None
        if isinstance(self.similarity, pd.DataFrame):
            self.similarity = self.similarity.values
        sim_matrix = self.similarity

        for i, s in enumerate(scales):
            df.at[i, "Magnitude"] = magnitude_of_matrix(sim_matrix ** s, method=self.inv_method)
        
        return df
    
    def plot_magnitude_curve(self, scales: t.Union[t.List[float], np.ndarray]= np.linspace(0.01, 10.0, 50), verbose: bool = False) -> Figure:
        df = self.magnitude_curve(scales)
        if verbose:
            print(df)

        plt.plot(df["Scale"], df["Magnitude"], marker='o')
        plt.title("Magnitude Curve")
        plt.xlabel("Scale")
        plt.ylabel("Magnitude")
        plt.axhline(y = int(self.points), color = 'r', linestyle = '--', label="Number of points")
        # plt.xscale("log")
        plt.grid()

        return plt.gcf()
    
    def plot_graph(self, **kargs) -> Figure:
        assert self.labels is not None
        assert self.similarity is not None
        if isinstance(self.similarity, pd.DataFrame):
            self.similarity = self.similarity.values
        return plot_graph(self.labels, self.similarity.tolist(), **kargs)


def compute_magnitude_curve(
        size: int = 10,
        buffer: int = 3,
        labels: t.Optional[t.List[str]] = None, 
        scales: t.Optional[t.List[float]] = None
    ) -> None:
    if labels is None:
        labels = random_language(size=size, buffer=buffer)
    assert labels is not None

    if scales is None:
        scales = np.linspace(0.01, 10.0, 50).tolist()
        # scales = np.logspace(-4, 4, 50).tolist()
    assert scales is not None

    model = MetricSpace(labels)
    print(model)

    model.plot_magnitude_curve(scales)

def test_invariance_of_magnitude(verbose: bool = True):
    labels: t.List[str] = random_language(size=10, buffer=3)
    model = MetricSpace(labels)

    if verbose:
        print("-> Modelo original:")
        print(model)
        print("\n")
        print("Iniciando testes de invariância da magnitude...\n")
        
    np.random.seed(42)
    # Embaralha a linguagem
    for i in range(5):
        shuffled = labels.copy()
        np.random.shuffle(shuffled)

        model2 = MetricSpace(shuffled)

        if verbose:
            print()
            print(model2)
            print(f"[Teste {i=}] Magnitude original: {model.get_magnitude():.6f}, embaralhada: {model2.get_magnitude():.6f}")
        assert np.isclose(model.get_magnitude(), model2.get_magnitude()), "Magnitude mudou após embaralhar!"

def test_robustness_of_magnitude_to_sampling_language(size: int = 10, buffer: int = 3, seeds_size: int = 5):
    seeds = range(seeds_size)
    magnitudes = [0.0 for _ in seeds]
    for i,seed in enumerate(seeds):
        labels = random_language(size, buffer, seed)
        model = MetricSpace(labels)
        magnitudes[i] = model.get_magnitude()

    print("\nMagnitudes of example models:")
    df = pd.DataFrame({
        "Seed": seeds,
        "Magnitude": magnitudes
    }).set_index("Seed")

    print(df)
    print(f"\nMean Magnitude: {np.mean(magnitudes):.6f} ± {np.std(magnitudes):.6f}")

def print_example_model(size: int = 10, buffer: int = 3):
    labels = random_language(size, buffer)
    self = MetricSpace(labels)
    print(self)

    assert self.similarity is not None
    if isinstance(self.similarity, pd.DataFrame):
        self.similarity = self.similarity.values
    plot_graph(labels, self.similarity.tolist()).savefig("figures/language_graph.png")

def benchmark_metric_spaces_methods(
        methods: t.Dict[str, MatrixInverter] = INVERSION_METHODS
    ) -> None:
    sizes = [100, 140, 150, 160, 170]
    # sizes = range(149, 152+1)
    buffer = 4 #len(ALPHABET)  # maximum buffer

    means = pd.DataFrame({
        str(size): [0.0 for _ in methods.keys()] for size in sizes
    }, index = list(methods.keys()))
    
    for size in sizes:
        labels = random_language(size, buffer)
        model = MetricSpace(labels)
        # print(f"{model.points} points in the metric space.")

        print(f"\n | Benchmarking magnitude calculation methods for {size=}")
        assert model.similarity is not None
        df = benchmark_magnitude_methods(model.similarity, repeat=5, methods=methods)
        means[str(size)] = df["Mean"]

    print("\n================================================")
    print(" Mean times (s) for each method and size:")
    print(means)
    print("================================================")

def test_magnitude_sensibility_to_buffer():
    seed = 42
    size = 12

    buffers = list(range(2, 10))
    magnitudes = [0.0 for _ in buffers]

    for i,buffer in enumerate(buffers):
        labels = random_language(size=size, buffer=buffer, seed=seed)
        # labels = generate_all_sentences(size, max_N=buffer)

        model = MetricSpace(labels)
        magnitudes[i] = model.get_magnitude()

        print(f"[Buffer={buffer}]")
        print(model)

    # display magnitudes
    for buffer, mag in zip(buffers, magnitudes):
        print(f"Buffer size: {buffer}, Magnitude: {mag:.6f}")

def test_magnitude_for_different_mass_functions(mass_functions: t.Dict[str, MassFunction] = MASS_FUNCTIONS):
    seed = 42
    size = 7
    buffer = 4

    labels = random_language(size=size, buffer=buffer, seed=seed)

    magnitudes = pd.DataFrame({
        "Function": list(mass_functions.keys()),
        "Magnitude": [0.0 for _ in mass_functions.keys()]
    }).set_index("Function")

    for name, fun in mass_functions.items():
        model = MetricSpace(labels, scale=1.0, mass_func=fun)
        print(f"[Mass Function: {name}]")
        print(model)

        magnitudes.at[name, "Magnitude"] = model.get_magnitude()

    print("\nMagnitudes for different mass functions:")
    print(magnitudes)

if __name__ == "__main__":
    print_example_model(size=10, buffer=4)

    # test_magnitude_for_different_mass_functions()
    # test_magnitude_sensibility_to_buffer()
