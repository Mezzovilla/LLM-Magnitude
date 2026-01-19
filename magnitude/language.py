from itertools import product, groupby
import typing as t
import numpy as np

# ALPHABET = ['eu', 'amo', 'te', 'você'] #, 'ela', 'ele', 'nós', 'vocês', 'eles', 'elas']
ALPHABET = ['a', 'b', ]#'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
START = '⟨'
END   = '⟩'
BUFFER = 5

def is_extension(x:str, y: str) -> bool:
    """
    Retorna True se y é extensão de x.
    Isso significa: x é prefixo de y.
    """
    return y.startswith(x)

def generate_all_sentences(
        N: int,
        max_sentences: int = 5_000_000,
        max_N: int = 20,
        alphabet: t.List[str] = ALPHABET,
        split_token: str = "|",
        start: str = START,
        end: str = END
    ) -> t.List[str]:
    """
    Gera todas as sentenças possíveis, completas ou incompletas, de comprimento
    máximo `N`, construídas a partir de um alfabeto dado.

    Cada sentença é formada concatenando:
        - o marcador inicial `⟨`,
        - zero ou mais símbolos do `alphabet`,
        - opcionalmente o marcador final `>`.

    Assim, para cada comprimento k ∈ {0, …, N}, a função produz:
        - uma sentença incompleta: ⟨ + (símbolos)  
        - uma sentença completa:   ⟨ + (símbolos) + >

    O total estimado de sentenças geradas é:
        2 * (1 + A + A² + ... + Aᴺ),
    onde A = len(alphabet).  
    Para evitar explosão combinatória, limites superiores são impostos.

    Parameters
    ----------
    N : int
        Comprimento máximo da parte intermediária (símbolos do alfabeto).
    max_sentences : int, optional
        Número máximo permitido de sentenças geradas.  
        A função aborta se a estimativa exceder esse limite.
    max_N : int, optional
        Maior valor permitido para `N`.  
        Evita escolha inadvertida de valores que causem explosão combinatória.
    alphabet : list[str], optional
        Lista de símbolos usados para gerar as sentenças.

    Returns
    -------
    list[str]
        Lista ordenada com todas as sentenças geradas, completas ou não.

    Raises
    ------
    ValueError
        Se `N` exceder `max_N`, ou se a estimativa de número de sentenças
        ultrapassar `max_sentences`.
    RuntimeError
        Se, por alguma razão, a geração ultrapassar `max_sentences`
        durante o processo (degradação inesperada).

    Examples
    --------
    >>> generate_all_sentences(1, alphabet=["a", "b"])
    [
        "⟨",
        "⟨⟩",
        "⟨a",
        "⟨a⟩",
        "⟨b",
        "⟨b⟩",
    ]
    """
    if N > max_N:
        raise ValueError(
            f"N={N} excede o máximo permitido max_N={max_N}. "
            "Abortando para evitar explosão combinatória."
        )

    # calcula uma estimativa antes de gerar
    A = len(alphabet)

    estimated = 2 * ((A ** (N+1) - 1) // (A - 1)) if A != 1 else 2 * (N + 1)

    if estimated > max_sentences:
        raise ValueError(
            f"Seriam geradas {estimated:,} sentenças, "
            f"acima do limite max_sentences={max_sentences:,}. "
            "Use um N menor."
        )
    
    sentences = set()
    count = 0

    for k in range(N+1):
        for middle in product(alphabet, repeat=k):
            mid = split_token.join(middle)

            s_incomplete = start + mid
            sentences.add(s_incomplete)

            s_full = start + mid + end
            sentences.add(s_full)

            count += 2
            if count > max_sentences:
                raise RuntimeError(
                    f"Exited early: threshold max_sentences={max_sentences} atingido!"
                )

    return sorted(sentences)

def random_language(
        size: int, 
        buffer: int, 
        seed: int = 42, 
        alphabet: t.List[str] = ALPHABET,
        split_token: str = "|",
        start: str = START,
        end: str = END
    ) -> t.List[str]:
    """
    Gera uma linguagem aleatória composta por `size` sentenças distintas,
    escolhidas sem reposição dentre todas as sentenças possíveis com
    comprimento intermediário máximo igual a `buffer`.

    A função utiliza `generate_all_sentences` para construir o conjunto
    completo de sentenças possíveis — completas ou incompletas — obedecendo
    ao alfabeto fornecido. Em seguida, realiza uma amostragem uniformemente
    aleatória (sem reposição) e retorna o subconjunto selecionado em ordem
    lexicográfica.

    A semente (`seed`) é utilizada para garantir reprodutibilidade da
    seleção aleatória.

    Parameters
    ----------
    size : int
        Número de sentenças a serem sorteadas na linguagem resultante.
        Deve ser menor ou igual ao total de sentenças geradas por
        `generate_all_sentences(buffer, alphabet)`.
    buffer : int
        Comprimento máximo da parte intermediária das sentenças
        (mesmo parâmetro `N` da função `generate_all_sentences`).
    seed : int, optional
        Semente do gerador de números aleatórios usada para garantir que
        a seleção seja determinística. O padrão é 42.
    alphabet : list[str], optional
        Lista de símbolos permitidos na construção das sentenças.

    Returns
    -------
    list[str]
        Uma lista ordenada, de tamanho `size`, contendo sentenças distintas
        escolhidas aleatoriamente do universo gerado.

    Raises
    ------
    ValueError
        Caso `size` exceda o número total de sentenças disponíveis.
        (Esse erro virá do NumPy ao tentar sortear mais elementos do que há.)

    Notes
    -----
    - Cada sentença segue o formato definido em `generate_all_sentences`:
      inicia com `⟨`, possui entre 0 e `buffer` símbolos do alfabeto,
      e pode ou não encerrar com `⟩`.
    - A amostragem é uniforme e sem reposição.

    Examples
    --------
    >>> random_language(3, buffer=1, seed=0, alphabet=["a"])
    ['⟨', '⟨a', '⟨a⟩']
    """
    np.random.seed(seed)
    all_sentences: t.List[str] = generate_all_sentences(buffer, 
        alphabet = alphabet,
        start = start,
        end = end,
        split_token=split_token
    )
    
    # get a sample from all_sentences but all elements must start with START and have no size greater than buffer
    chosen = np.random.choice(all_sentences, size=size, replace=False).tolist()
    return sorted(chosen)