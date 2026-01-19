import torch
import torch.nn as nn
import torch.nn.functional as F
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

LEFT_SPACE = 'Ġ'

from dataclasses import dataclass, field

@dataclass(slots=True)
class PreTrainedLM():
    model: t.Union[GPT2LMHeadModel, nn.Module]
    tokenizer: GPT2TokenizerFast
    complete_phrases: t.List[str]
    phrases: t.List[str] = field(init=False)
    tokens: t.List[str] = field(init = False)
    vocab_size: int = field(init=False)

    def get_tokens(self, 
            phrases: t.Optional[t.Union[t.List[str], str]] = None, 
            sort: bool=True
        ) -> t.List[str]:
        tokens = []

        if phrases is None:
            phrases = self.phrases

        if isinstance(phrases, str):
            phrases = [phrases]
        
        for phrase in phrases:
            for tok in self.tokenizer.tokenize(phrase):
                tokens.append(tok)
        tokens = list(dict.fromkeys(tokens))  # remove duplicates, preserve order
        return sorted(tokens) if sort else tokens

    def from_complete_sentences(self, verbose: bool = False) -> t.List[str]:
        phrases = []

        for sentence in self.complete_phrases:
            tokens = self.tokenizer.tokenize(sentence)

            for i in range(len(tokens)):
                phrases.append("".join(tokens[:i+1]).replace(LEFT_SPACE, ' '))

        return phrases

    def __post_init__(self):
        if not isinstance(self.complete_phrases, list):
            raise ValueError('Phrases must be a list of strings.')
        
        self.tokens = self.get_tokens(self.complete_phrases)
        self.phrases = self.from_complete_sentences() 
        self.vocab_size = len(self.phrases)

    def token_probability(self, y_token: str, x: str, verbose: bool = False) -> float:
        '''p(a | x), a in {tokens}, x in {tokens}*'''
        input_ids = self.tokenizer(x, return_tensors="pt").input_ids

        out = self.model(input_ids)
        logits = out.logits[0, -1]

        y_id = self.tokenizer.encode(y_token, add_special_tokens=False)[0]
        probs = 1/(1 + np.exp(-logits[y_id].item()))

        if verbose:
            print(f"p({y_token=} | {x=}) = {probs}".replace(LEFT_SPACE, " "))
        return probs


    def extension_probability(
            self, 
            y: str, 
            x: str, 
            verbose: bool = False, 
        ) -> float:
        '''pi(y | x) = prod p(yi | xy[0..i-1]),  y, x in {tokens}*'''
        if not y.startswith(x):
            return 0.0
        
        if x == y:
            return 1.0

        x_tokens = self.get_tokens(x, sort=False)
        y_tokens = self.get_tokens(y, sort=False)

        new_tokens = [y_tokens[i] for i in range(len(x_tokens), len(y_tokens))]
        
        probs = [0.0 for _ in new_tokens]
        xt = x
    
        for i, yt in enumerate(new_tokens):
            probs[i] = self.token_probability(yt, xt, verbose)
            xt += yt

        pi = float(np.prod(probs))

        if verbose:
            print(f"{probs=}")
            print(f"pi({y=}|{x=}) = {pi}")

        return pi

    def similarity_matrix(self, phrases: t.Optional[t.List[str]] = None) -> pd.DataFrame:
        if phrases is None:
            phrases = self.phrases
        phrases = list(dict.fromkeys(phrases))  # remove duplicates, preserve order
        sim_matrix = np.zeros((len(phrases), len(phrases)))

        for i, y_phrase in enumerate(phrases):
            for j, x_phrase in enumerate(phrases):
                sim_matrix[i, j] = self.extension_probability(x_phrase, y_phrase)
        return pd.DataFrame(sim_matrix, index = phrases, columns=phrases)
    
    def metric_space(self, phrases: t.Optional[t.List[str]] = None, **kargs) -> MetricSpace:
        if phrases is None:
            phrases = self.phrases
        return MetricSpace(similarity=self.similarity_matrix(phrases), **kargs)

    def lm_magnitude(self,
            phrases: t.Optional[t.List[str]] = None,
            verbose: bool = False
        ) -> float:
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
        if verbose:
            print(":: Similarity matrix:")
            print(df_sim)
        
        # Precompute "length without spaces"
        lengths = {
            p: len(self.get_tokens(p))
            for p in phrases
        }
        if verbose:
            print(f":: Lengths (without spaces): {lengths}")


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

                    if verbose:
                        print(f":: Adding P({y=} | {prefix=}) = {df_sim[y][prefix]}")
                        print(f"   Current sum: {df_sim.shape[0] - magnitude_upper_sum}")

        return df_sim.shape[0] - magnitude_upper_sum

MODELS = {
    # "not": "danurahul/alex_gpt3_Doctextfull2",
    # "Norod78/hewiki-articles-distilGPT2py-il",
    'distil_gpt2': "Norod78/english-sienfeld-distilgpt2",
}

def pretrained_model():
    project_path = Path(__file__).parent

    checkpoint = "Norod78/english-sienfeld-distilgpt2"
    model_path = project_path / f"models/{checkpoint}"
    print(':: Reading model')

    tokenizer = (AutoTokenizer
        .from_pretrained(checkpoint, local_files_only=True)
    )
    model = (AutoModelForCausalLM
        .from_pretrained(checkpoint, local_files_only=True)
    )

    print(f":: Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB")

    complete_phrases = [
        # "The cat sleeps in the living room",
        # "The cat runs in the living room",
        # "The dog runs in the yard",
        "The dog runs",
        "The cat sleeps",
        # "The cockroach flies in the house"
    ]

    pt_model = PreTrainedLM(model, tokenizer, complete_phrases)

    # pt_model.extension_probability(x='The cat', y='The cat runs in the living room', verbose=True)

    # pt_model.token_probability(y_token=LEFT_SPACE+'runs', x='The dog', verbose=True)

    print(pt_model.similarity_matrix())
    # print(pt_model.lm_magnitude())
    # print(pt_model.metric_space())

if __name__ == '__main__':
    pretrained_model()
