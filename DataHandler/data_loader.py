import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

def load_split_raw():
    # Use original texts (Uncleaned)
    df_imdb_tr = pd.read_csv("../raw_data/splits/imdb_train.csv")
    df_imdb_dev = pd.read_csv("../raw_data/splits/imdb_dev.csv")
    df_imdb_te = pd.read_csv("../raw_data/splits/imdb_test.csv")

    df_douban_tr = pd.read_csv("../raw_data/splits/douban_train.csv")
    df_douban_dev = pd.read_csv("../raw_data/splits/douban_dev.csv")
    df_douban_te = pd.read_csv("../raw_data/splits/douban_test.csv")

    # Mixed
    df_tr = pd.concat([df_imdb_tr, df_douban_tr], ignore_index=True)
    df_dev = pd.concat([df_imdb_dev, df_douban_dev], ignore_index=True)
    df_te = pd.concat([df_imdb_te, df_douban_te], ignore_index=True)

    return (
        (df_tr["review"].tolist(), df_tr["label"].tolist(), df_tr["source"].tolist()),
        (df_dev["review"].tolist(), df_dev["label"].tolist(), df_dev["source"].tolist()),
        (df_te["review"].tolist(), df_te["label"].tolist(), df_te["source"].tolist()),
    )

class Word2VecDataset(Dataset):
    def __init__(self, tokenized_sentences, labels, sources, w2v_model):
        dim = w2v_model.vector_size
        embeddings = []
        for tokens in tokenized_sentences:
            vectors = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
            if vectors:
                arr = np.stack(vectors, axis=0).astype(np.float32)
                embeddings.append(arr.mean(axis=0))
            else:
                embeddings.append(np.zeros(dim, dtype=np.float32))

        self.embeddings = torch.from_numpy(np.stack(embeddings, axis=0))
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.sources = sources

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "x": self.embeddings[idx],
            "labels": self.labels[idx],
            "source": self.sources[idx],
        }

class FeatureDataset(Dataset):
    def __init__(self, X, y, sources):
        # X: numpy array or sparse matrix, y: list or array
        self.X = torch.tensor(X.toarray() if hasattr(X, "toarray") else X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.sources = sources
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "labels": self.y[idx],
            "source": self.sources[idx],
        }

# BertDataset: During initialization, all samples are passed through fixed mBERT (using torch.no_grad()), and the [CLS] vector for each text is cached in self.embeddings.
# In the subsequent training, validation, and testing loops, the DataLoader retrieves static vectors directly from this cache via __getitem__, without touching BERT again, making it extremely fast!
# But initially made a mistake: only loaded the training set, and saved all the training CLS vectors into cache/bert_embs.pt.
# Then mistakenly constructed the dev/test sets using this training cache, and "read" training vectors as dev/test vec.
# The labels didn’t align, so the results ended up at random guessing level—around ~50%.

# Why use cached embeddings?
# BertClassifier with freeze_bert=True:
# Although all BERT parameters are set to requires_grad=False to disable gradient computation and backpropagation, each batch still requires a full forward pass through BERT using the input input_ids and attention_mask.
# Even when wrapped in torch.no_grad(), the computational cost remains similar to standard fine-tuning, only gradients are not calculated.
# If using cached results, the outcomes are effectively the same!

def tokenize_batch(batch_text, tokenizer, max_length=128, return_tensors=None):
    return tokenizer(
        batch_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors = return_tensors
    )

class BertEmbedsDataset(Dataset):
    def __init__(self, texts, labels, sources, pretrained_model="bert-base-multilingual-uncased",
                 data_name=None, max_length=128, batch_size=32, device=None):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.sources = sources
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cache_path = f"cache/bert_embeddings_{data_name}.pt"

        # Time-consuming, cache
        if os.path.exists(cache_path):
            print(f"Loading cached BERT embeddings from {cache_path}")
            self.embeddings = torch.load(cache_path, map_location="cpu")
            return

        print("Cache not found, computing BERT embeddings…")

        # Initialize tokenizer + bert
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert = AutoModel.from_pretrained(pretrained_model).to(self.device)
        self.bert.eval()  # Freeze

        # Pre-calculate CLS vectors
        all_embeddings = []
        for i in range(0, len(self.texts), batch_size):
            batch = self.texts[i: i + batch_size]
            enc = tokenize_batch(batch, tokenizer=self.tokenizer, max_length=max_length, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out = self.bert(**enc)

            all_embeddings.append(out.last_hidden_state[:, 0].cpu())

        # Tensor shape=(N, hidden_size)
        self.embeddings = torch.cat(all_embeddings, dim=0)

        print(f"Saving {data_name} BERT embeddings to {cache_path}")
        torch.save(self.embeddings, cache_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "x": self.embeddings[idx],
            "labels": self.labels[idx],
            "source": self.sources[idx],
        }