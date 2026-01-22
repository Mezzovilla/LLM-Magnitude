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


LEFT_SPACE = 'Ġ'

from dataclasses import dataclass, field

@dataclass(slots=True)
class PreTrainedLM():
    model: t.Union[GPT2LMHeadModel, nn.Module]
    tokenizer: GPT2TokenizerFast
    complete_sentences: t.List[str]
    phrases: t.List[str] = field(init=False)
    tokens: t.List[str] = field(init=False)

    # Derived / internal
    sentence_ids: t.List[torch.Tensor] = field(init=False)
    prefix_ids: t.List[torch.Tensor] = field(init=False)
    num_prefixes: int = field(init=False)

    def encode(self, text: str) -> torch.Tensor:
        """Encode text into a 1D tensor of token ids (no special tokens)."""
        return self.tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]

    def decode(self, ids: torch.Tensor) -> str:
        """Decode token ids into text."""
        return self.tokenizer.decode(ids.tolist())

    def build_prefix_space(self) -> t.List[torch.Tensor]:
        """
        From complete sentences, build the set of all non-empty prefixes
        in token-id space.
        """
        prefixes: t.List[torch.Tensor] = []

        for ids in self.sentence_ids:
            for k in range(1, len(ids) + 1):
                prefixes.append(ids[:k])

        # remove duplicates while preserving order
        seen = set()
        unique_prefixes = []
        for p in prefixes:
            key = tuple(p.tolist())
            if key not in seen:
                seen.add(key)
                unique_prefixes.append(p)

        return unique_prefixes

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

        for sentence in self.complete_sentences:
            tokens = self.tokenizer.tokenize(sentence)

            for i in range(len(tokens)):
                phrases.append("".join(tokens[:i+1]).replace(LEFT_SPACE, ' '))

        return phrases

    def __post_init__(self):
        if not isinstance(self.complete_sentences, list):
            raise ValueError("complete_sentences must be a list of strings.")

        # Encode sentences once
        self.sentence_ids = [self.encode(s) for s in self.complete_sentences]

        # Build prefix space
        self.prefix_ids = self.build_prefix_space()
        self.num_prefixes = len(self.prefix_ids)


        # self.tokens = self.get_tokens(self.complete_sentences)
        self.phrases = self.from_complete_sentences()

    @torch.no_grad()
    def token_probability(
            self,
            y_id: int,
            x_ids: torch.Tensor,
            verbose: bool = False,
        ) -> float:
        """
        p(y | x), where:
          - y_id is a token id
          - x_ids is a 1D tensor of token ids
        """
        logits = self.model(x_ids.unsqueeze(0)).logits[0, -1]
        prob = F.softmax(logits, dim=-1)[y_id].item()

        if verbose:
            x_str = self.decode(x_ids)
            y_str = self.decode(torch.tensor([y_id]))
            print(f"p({y_str!r} | {x_str!r}) = {prob:.6f}")

        return prob


    @torch.no_grad()
    def extension_probability(
            self,
            y_ids: torch.Tensor,
            x_ids: torch.Tensor,
            verbose: bool = False,
        ) -> float:
        """
        π(y | x) = ∏ p(y_i | x y_1 ... y_{i-1})

        Returns:
          0 if x is not a prefix of y
          1 if x == y
        """
        if len(x_ids) > len(y_ids):
            return 0.0

        if not torch.equal(y_ids[: len(x_ids)], x_ids):
            return 0.0

        if len(x_ids) == len(y_ids):
            return 1.0

        probs = []
        xt = x_ids.clone()

        for yi in y_ids[len(x_ids):]:
            p = self.token_probability(int(yi.item()), xt, verbose)
            probs.append(p)
            xt = torch.cat([xt, yi.view(1)])

        pi = float(np.prod(probs))

        if verbose:
            print(f"probs = {probs}")
            print(f"π(y | x) = {pi}")

        return pi

    @property
    def similarity_matrix(self) -> pd.DataFrame:
        """
        Similarity matrix S where:
          S[i, j] = π(prefix_j | prefix_i)
        """
        n = len(self.prefix_ids)
        mat = np.zeros((n, n))

        for i, x in enumerate(self.prefix_ids):
            for j, y in enumerate(self.prefix_ids):
                mat[i, j] = self.extension_probability(y, x)

        labels = [self.decode(p) for p in self.prefix_ids]
        return pd.DataFrame(mat, index=labels, columns=labels)
    
    @property
    def metric_space(self, **kargs) -> MetricSpace:
        return MetricSpace(similarity=self.similarity_matrix, **kargs)

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

        df_sim = self.similarity_matrix
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

    checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model_path = project_path / f"models/{checkpoint}"
    print(':: Reading model')

    tokenizer = (AutoTokenizer
        .from_pretrained(checkpoint, local_files_only=True)
    )
    model = (AutoModelForCausalLM
        .from_pretrained(checkpoint, local_files_only=True)
    )

    print(f":: Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB")

    complete_sentences = [
        # "The cat sleeps in the living room",
        # "The cat runs in the living room",
        "The dog runs in the yard",
        "The dog doesn\'t sleeps in the yard",
        "The dog runs",
        # "The cat sleeps",
        # "The cockroach flies in the house"
    ]

    pt_model = PreTrainedLM(model, tokenizer, complete_sentences)

    # pt_model.extension_probability(x='The cat', y='The cat runs in the living room', verbose=True)

    # pt_model.token_probability(y_token=LEFT_SPACE+'runs', x='The dog', verbose=True)

    df= pt_model.similarity_matrix
    print(df)
    print(df.shape)
    # print(f"magnitude = {pt_model.lm_magnitude()}")
    ms_model = pt_model.metric_space
    curve = ms_model.magnitude_curve()
    
    curve.to_csv("magnitude_curve.csv")


if __name__ == '__main__':
    pretrained_model()
