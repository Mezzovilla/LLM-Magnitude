import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random
import numpy as np
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. Entropia de Tsallis
# ============================================================

def tsallis_entropy_from_logits(logits, k):
    """
    logits: tensor (V,)
    """
    probs = F.softmax(logits, dim=-1)
    sum_pk = torch.sum(probs ** k)
    return (1.0 / (k - 1.0)) * (1.0 - sum_pk)


# ============================================================
# 2. Dataset simples de sentenças
# ============================================================

class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


# ============================================================
# 3. Regressor MLP
# ============================================================

class EntropyRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)


# ============================================================
# 4. Monitor principal
# ============================================================

class ComplexityMonitor:
    def __init__(
        self,
        model_name,
        sentences,
        k=2.0,
        monitor_size=10000,
        batch_size=16,
    ):
        self.k = k
        self.batch_size = batch_size

        print("Carregando modelo...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True
        ).to(device)
        self.model.eval()

        # Seleciona conjunto M
        self.M = random.sample(sentences, min(monitor_size, len(sentences)))
        self.dataset = SentenceDataset(self.M)

        self.regressor = None

    # --------------------------------------------------------
    # Extração de embedding
    # --------------------------------------------------------
    def get_embedding(self, sentence):
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Última camada oculta
        hidden_states = outputs.hidden_states[-1]  # (1, seq_len, d_model)

        # Média dos tokens
        embedding = hidden_states.mean(dim=1).squeeze(0)  # (d_model,)
        return embedding

    # --------------------------------------------------------
    # Cálculo exato de H_k
    # --------------------------------------------------------
    def compute_exact_entropy(self, sentence):
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[:, -1, :]  # último token
        return tsallis_entropy_from_logits(logits.squeeze(0), self.k)

    # --------------------------------------------------------
    # Construção do dataset de treino do regressor
    # --------------------------------------------------------
    def build_regression_dataset(self, sample_size=None):
        print("Construindo dataset de regressão...")

        if sample_size:
            subset = random.sample(self.M, sample_size)
        else:
            subset = self.M

        embeddings = []
        entropies = []

        for sentence in tqdm(subset):
            emb = self.get_embedding(sentence)
            H = self.compute_exact_entropy(sentence)

            embeddings.append(emb.cpu())
            entropies.append(H.cpu())

        X = torch.stack(embeddings)
        y = torch.stack(entropies)

        return X, y

    # --------------------------------------------------------
    # Treinamento do regressor
    # --------------------------------------------------------
    def train_regressor(self, epochs=20, lr=1e-3):
        X, y = self.build_regression_dataset(sample_size=min(2000, len(self.dataset)))

        input_dim = X.shape[1]
        self.regressor = EntropyRegressor(input_dim).to(device)

        optimizer = torch.optim.Adam(self.regressor.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        X = X.to(device)
        y = y.to(device)

        print("Treinando regressor...")

        for epoch in range(epochs):
            self.regressor.train()

            optimizer.zero_grad()
            preds = self.regressor(X)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")

    # --------------------------------------------------------
    # Estimativa rápida da soma
    # --------------------------------------------------------
    def estimate_complexity(self, total_X_size):
        assert self.regressor is not None, "Regressor não treinado"

        self.regressor.eval()

        embeddings = []
        for sentence in tqdm(self.M):
            emb = self.get_embedding(sentence)
            embeddings.append(emb.cpu())

        X = torch.stack(embeddings).to(device)

        with torch.no_grad():
            H_hat = self.regressor(X)

        mean_entropy = H_hat.mean().item()

        estimated_sum = total_X_size * mean_entropy
        return estimated_sum

    # --------------------------------------------------------
    # Validação periódica
    # --------------------------------------------------------
    def validate(self, sample_size=None):
        new_sample_size = int(min(100, len(self.dataset)))

        if sample_size is None:
            sample_size = new_sample_size

        if sample_size > len(self.dataset):
            print(f"Sample size is bigger than dataset. Changing: {sample_size} ==> {new_sample_size}")
            sample_size = new_sample_size

        subset = random.sample(self.M, sample_size)

        exact = []
        predicted = []

        self.regressor.eval()

        for sentence in subset:
            emb = self.get_embedding(sentence).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = self.regressor(emb).item()

            H = self.compute_exact_entropy(sentence).item()

            exact.append(H)
            predicted.append(pred)

        exact = np.array(exact)
        predicted = np.array(predicted)

        mse = np.mean((exact - predicted) ** 2)
        print(f"MSE validação: {mse:.6f}")
        return mse

def main():
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
    # Corpus de exemplo
    sentences = [
        "O gato está sobre a mesa.",
        "A matemática aplicada é fascinante.",
        "Transformers revolucionaram o NLP.",
    ]

    monitor = ComplexityMonitor(
        model_name=checkpoint,
        sentences=sentences,
        k=2.0,
        monitor_size=5000
    )

    monitor.train_regressor(epochs=15)

    # Estimar C(kX_t)
    total_size = len(sentences)
    complexity_estimate = monitor.estimate_complexity(total_size)
    print("Estimativa da soma:", complexity_estimate)

    # Validar
    monitor.validate(sample_size=100)

if __name__ == "__main__":
    main()
