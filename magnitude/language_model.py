import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import typing as t
import matplotlib.pyplot as plt
from rich import print
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.auto.modeling_auto import _BaseModelWithGenerate

from magnitude.metric_space import MetricSpace
from magnitude.language import (
    generate_all_sentences,
    random_language
)
torch.manual_seed(0)

TOKENS = ["A", "B", "<EOS>"]
SPLITTING_CHAR = "|"

def _get_tokens(phrases: t.List[str]) -> t.List[str]:
    tokens = []
    for p in phrases:
        if len(p) == 0:
            continue

        for t in p.split(' '):
            if t not in tokens:
                tokens.append(t)
    tokens.append("<EOS>")
    return tokens

def next_token_dataset(
        phrases: t.List[str], 
        tokens: t.Optional[t.List[str]] = None
    ) -> t.Tuple[t.List[str], t.List[str]] :
    contexts, targets = [], []

    if tokens is None:
        tokens = _get_tokens(phrases)

    for p in phrases:
        # if p == "":
        #     context = []
        # else:
        #     context = p.split()

        for t in tokens:
            candidate = (p + " " + t).strip()

            if candidate in phrases:
                contexts.append(p)
                targets.append(t)

    return contexts, targets


class ARLM(nn.Module):
    def __init__(self,
            phrases: t.List[str],
            tokens: t.Optional[t.List[str]] = None,
        ):
        super().__init__()
        
        if not isinstance(phrases, list):
            raise ValueError()
        
        self.phrases = phrases

        if tokens is None:
            tokens = _get_tokens(phrases)
        self.tokens = tokens
        
        vocab_size = len(self.tokens)
        self.linear = nn.Linear(vocab_size, vocab_size, bias=False)

    def token_to_id(self, tokens: t.Optional[t.List[str]] = None) -> t.Dict[str, int]:
        if tokens is None:
            tokens = self.tokens
        return {t: i for i, t in enumerate(tokens)}
    
    def id_to_token(self, tokens: t.Optional[t.List[str]] = None) -> t.Dict[int, str]:
        return {i: t for t, i in self.token_to_id(tokens).items()}

    def encode_context(
            self,
            context: str, 
            tokens: t.Optional[t.List[str]] = None,
        ) -> torch.Tensor:
        if tokens is None:
            tokens = self.tokens
        
        token_to_id = self.token_to_id(tokens)

        vec = torch.zeros(len(tokens))
        if context == "":
            return vec

        for t in context.split():
            vec[token_to_id[t]] += 1
        return vec

    def forward(self, x):
        return self.linear(x)

    def extension_probability(self, y: str, x: str, 
            tokens: t.Optional[t.List[str]] = None,
            verbose: bool = False, 
        ) -> float:
        if not y.startswith(x):
            return 0.0
        
        if x == y:
            return 1.0

        x_tokens = x.split() if x != "" else []
        y_tokens = y.split() if y != "" else []

        new_tokens = [y_tokens[i] for i in range(len(x_tokens), len(y_tokens))]
        
        if tokens is None:
            tokens = self.tokens
        token_to_id: t.Dict[str, int] = self.token_to_id(tokens) 
        
        if verbose:
            print(f"{x=}, {y=}, {new_tokens=}")

        probs = [0.0 for _ in new_tokens]
        xt = x

        encoder = lambda x: self.encode_context(x, tokens)
        for i, yt in enumerate(new_tokens):
            probs[i] = torch.softmax(self(encoder(xt)), dim = 0)[token_to_id[yt]].item()
            xt += " " + yt

        pi = float(np.prod(probs))

        if verbose:
            print(f"{probs=}")
            print(f"pi({y=}|{x=}) = {pi}")

        return pi

    def similarity_matrix(self, phrases: t.Optional[t.List[str]] = None) -> pd.DataFrame:
        if phrases is None:
            phrases = self.phrases
        S = np.zeros((len(phrases), len(phrases)))

        for i, y_phrase in enumerate(phrases):
            for j, x_phrase in enumerate(phrases):
                S[i, j] = self.extension_probability(x_phrase, y_phrase)
        return pd.DataFrame(S, index = phrases, columns=phrases)
    
    def metric_space(self, phrases: t.Optional[t.List[str]] = None) -> MetricSpace:
        if phrases is None:
            phrases = self.phrases
        return MetricSpace(similarity=self.similarity_matrix(phrases))
    
    def lm_magnitude(self, phrases: t.Optional[t.List[str]] = None) -> float:
        """
        Compute:
            df_sim.shape[0] - sum_x sum_{y extends x} P(y | x)

        where y extends x iff:
            - y.startswith(x)
            - len(y.replace(" ", "")) = len(x.replace(" ", "")) + 1

        Parameters
        ----------
        df_sim : pd.DataFrame
            Similarity matrix with phrases as index and columns.

        Returns
        -------
        float
        """
        if phrases is None:
            phrases = self.phrases

        df_sim = self.similarity_matrix(phrases)
        
        # Precompute "length without spaces"
        lengths = {
            p: len(p.split(" "))
            for p in phrases
        }

        # Group phrases by length
        by_length = defaultdict(list)
        for p, l in lengths.items():
            by_length[l].append(p)

        magnitude_upper_sum = 0.0

        for prefix in phrases:
            lp = lengths[prefix]

            candidates = by_length.get(lp + 1, [])

            for y in candidates:
                if y.startswith(prefix):
                    magnitude_upper_sum += df_sim[y][prefix]

        return df_sim.shape[0] - magnitude_upper_sum

    def train_model(self, 
            phrases: t.Optional[t.List[str]] = None,
            tokens: t.Optional[t.List[str]] = None,
            epochs: int = 100,
            learning_rate: float = 0.1,
            criterion = nn.CrossEntropyLoss(),
            verbose: bool = False
        ) -> t.Tuple[t.Self, pd.DataFrame]:
        if phrases is None:
            phrases = self.phrases
        if verbose:
            print(f"Phrases to train: ({len(phrases)})")
            print(phrases)
        columns = ["Loss", "Magnitude"]
        history = pd.DataFrame(
            np.zeros((epochs, len(columns))),
            columns = columns
        )

        if tokens is None:
            tokens = self.tokens
        
        contexts, targets = next_token_dataset(phrases, tokens)

        token_to_id: t.Dict[str, int] = self.token_to_id(tokens)

        X = torch.stack([self.encode_context(c, tokens) for c in contexts])
        y = torch.tensor([token_to_id[t] for t in targets])

        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()

            logits = self(X)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            history.loc[epoch] = [
                loss.item(), self.lm_magnitude(phrases)
            ]
        if verbose:
            # print(f"{tokens=}")
            # print(f"{contexts=}")
            # print(f"{targets=}")
            print(history)
        return (self, history)

def main():
    phrases = generate_all_sentences(4, alphabet=['A','B','C'], split_token=' ', start = '', end = '')

    model = ARLM(phrases)

    print(model.metric_space())

    model.train_model(verbose=True)
    
    print(model.metric_space())

if __name__ == '__main__':
    main()
