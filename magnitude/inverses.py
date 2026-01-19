import numpy as np
import typing as t
import scipy.linalg.lapack as lp
import scipy.linalg as la

def inverse_tri_geometric(T: np.ndarray) -> np.ndarray:
    """
    Calcula a inversa de T = I + N para uma matriz triangular superior
    representada por um pandas.DataFrame.

    Fórmula: (I + N)^{-1} = I + sum_{k=1}^{n-1} (-1)^k N^k
    """
    n = T.shape[0]

    I = np.eye(n)

    N = T - I

    # Inicializa inversa com I
    T_inv = I.copy()

    # Potência inicial de N
    Nk = N.copy()

    # Soma os termos N, N^2, ..., N^{n-1}
    for k in range(1, n):
        T_inv += ((-1)**k) * Nk
        Nk = Nk @ N  # proximo Nk = N^(k+1)

    return T_inv

# def invert_upper_triangular_lapack(T: np.ndarray) -> np.ndarray:
#     """
#     Inverte uma matriz triangular superior T usando LAPACK (dtrtri).
#     T deve ser um pd.DataFrame representando uma matriz triangular superior
#     com diagonal não nula (ex. diagonal = 1).
#     """
#     # Chamada ao LAPACK
#     # UPLO='U' (triangular superior)
#     # DIAG='N' (diagonal não-unidade — LAPACK calculará normalmente)
#     T_inv, info = lp.dtrtri(T, lower=0, unitdiag=0)

#     if info < 0:
#         raise ValueError(f"dtrtri: argumento {-info} inválido.")
#     elif info > 0:
#         raise np.linalg.LinAlgError("Matriz triangular é singular e não pode ser invertida.")
#     return T_inv

def test_inverse_upper_triangular(inverse_fun: t.Callable[[np.ndarray], np.ndarray]):
    # Exemplo de matriz triangular superior
    T = np.array([
        [1, 2, 3],
        [0, 1, 4],
        [0, 0, 1]
    ])

    T_inv = inverse_fun(T)

    # Verifica se T * T_inv é a matriz identidade
    identity = T @ T_inv
    expected_identity = np.eye(3)

    assert np.allclose(identity, expected_identity), "Inversa calculada está incorreta."

def invert_triangular(T: np.ndarray, upper: bool = True, unit_diagonal: bool = True) -> np.ndarray:
    I = np.eye(T.shape[0], dtype=T.dtype)
    return la.solve_triangular(T, I, lower=not upper, unit_diagonal=unit_diagonal, check_finite=False)

MatrixInverter = t.Callable[[np.ndarray], np.ndarray]

INVERSION_METHODS: t.Dict[str, MatrixInverter] = {
    # "geometric": inverse_tri_geometric, 
    "direct": lambda P: np.linalg.inv(P), 
    "pseudo": lambda P: np.linalg.pinv(P), 
    # "lapack": invert_upper_triangular_lapack, 
    "scipy-inv": lambda P: la.inv(P, check_finite=False),
    "scipy-pinv": lambda P: la.inv(P, check_finite=False),
    "scipy-tri-inv": lambda P: invert_triangular(P, upper=True, unit_diagonal=True)
}

def test_all_inversion_methods(methods: t.Dict[str, MatrixInverter] = INVERSION_METHODS):
    for name, func in methods.items():
        print(f" | Testando método de inversão: {name}")
        test_inverse_upper_triangular(func)

if __name__ == "__main__":
    test_all_inversion_methods()