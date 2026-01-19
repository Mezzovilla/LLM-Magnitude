import pandas as pd
import numpy as np
from magnitude.language import (
    is_extension,
    START,
    END
)
from magnitude.display import (
    display_pd
)
import typing as t
from functools import partial
from rich import print
# import Levenshtein



def freq_probabilities(y: str, x: str, language: t.List[str]) -> float:
    """
    Probabilidade de y ser uma extensão de x.
    Aqui uso regra simples:
      pi(y|x) = 1 / (# extensões possíveis de x)
    se y estende x, e 0 caso contrário.
    """
    if y == x:
        return 1.0

    if x.endswith(END):
        return 0.0
    
    if not is_extension(x, y):
        return 0.0

    size = len([z for z in language if is_extension(x, z)])

    if size == 0:
        return 0.0
    
    return 1.0 / size

def softmax(y: str, x: str, language: t.List[str], score: t.Callable[[str,str], float]) -> float:
    """
    Probabilidade de y ser uma extensão de x.
    Aqui uso uma função softmax baseada no comprimento das strings:
      pi(y|x) = exp(s(x,y)) / sum_{z} exp(s(x,y))
    se y estende x, e 0 caso contrário.
    """
    if y == x:
        return 1.0
    
    if x.endswith(END):
        return 0.0
    
    if not is_extension(x, y):
        return 0.0
    
    sx = partial(score, x)

    exp_scores = np.array([np.exp(sx(z)) for z in language])

    j: int = language.index(y)
    return exp_scores[j] / exp_scores.sum()


def length_score(y: str, x: str) -> float:
    ly = len(y)
    lx = len(x)

    if is_extension(x, y):
        return float(ly - lx)
    return 0.0

# def levenshtein_score(y: str, x: str) -> float:
#     if is_extension(x, y):
#         return float(-Levenshtein.distance(x, y))
#     return 0.0

def prefix_similarity_score(x: str, y: str, language: t.List[str] = ['']) -> int:
    """
    Retorna o tamanho do maior prefixo comum entre x e y.
    """
    n = min(len(x), len(y))
    i = 0
    while i < n and x[i] == y[i]:
        i += 1
    return i

def dirichlet_mass(x: str, y: str, language: t.List[str]) -> float:
    if not is_extension(y, x):
        return 0.0
    return float(np.random.dirichlet((1,3), size = len(language))[language.index(y)][0])


MassFunction = t.Callable[[str, str, t.List[str]], float]

MASS_FUNCTIONS: t.Dict[str, MassFunction] = {
    "freq_probabilities": freq_probabilities,
    "softmax_length": partial(softmax, score=length_score),
    # "softmax_levenshtein": partial(softmax, score=levenshtein_score),
    "softmax_prefix_sim": partial(softmax, score=prefix_similarity_score),
    "dirichlet_mass": dirichlet_mass
}

def build_P(
        language: t.List[str], 
        scale: float = 1,
        mass_func: t.Callable[[str, str, t.List[str]], float] = freq_probabilities
    ) -> np.ndarray:
    """
    Constrói matriz P = [ π(x_j | x_i) ]_{i,j}
    """
    n = len(language)
    P = np.eye(n)

    for i, xi in enumerate(language):
        for j, xj in enumerate(language):
            if i >= j:
                continue
            if xi.endswith(END):
                P[i,j] = 0.0
                continue

            P[i, j] = mass_func(xj, xi, language)**scale

    return P

def to_pandas(P: np.ndarray, language: t.Optional[t.List[str]] = None) -> pd.DataFrame:
    if language is None:
        language = [f'X{i}' for i in range(P.shape[0])]

    return pd.DataFrame(P, index = language, columns=language)

def test_probabilities_functions(functions: t.Dict[str, MassFunction] = MASS_FUNCTIONS):
    language = ['a', 'ab', 'bc', 'abcd']

    for name, func in functions.items():
        print(f"\n | Testando função de probabilidade: {name}")

        P = build_P(language, mass_func=func)
        print(display_pd(P, digits=3).to_string())  # type: ignore

if __name__ == "__main__":
    test_probabilities_functions()

