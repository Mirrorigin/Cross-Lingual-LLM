import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

def load_split_raw():
    # Use original texts (Uncleaned)
    df_imdb_tr = pd.read_csv("raw_data/splits/imdb_train.csv")
    df_imdb_dev = pd.read_csv("raw_data/splits/imdb_dev.csv")
    df_imdb_te = pd.read_csv("raw_data/splits/imdb_test.csv")

    df_douban_tr = pd.read_csv("raw_data/splits/douban_train.csv")
    df_douban_dev = pd.read_csv("raw_data/splits/douban_dev.csv")
    df_douban_te = pd.read_csv("raw_data/splits/douban_test.csv")

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
        # X: numpy array or sparse matrix，y: list or array
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

# BertDataset: 在初始化阶段一次性把所有样本跑过 BERT（with torch.no_grad()），把每条文本的 [CLS] 向量都缓存下来（self.embeddings
# 后面的训练、验证、测试循环中，DataLoader 直接从这个缓存里 __getitem__ 出一个静态的向量，不再触碰 BERT，速度非常快。
# 但这里我最开始犯了一个错误：只用来加载训练集，把 train 的所有 CLS 向量写进了 cache/bert_embs.pt。
# 然后直接拿 train cache 的向量构造 dev/test 集，“读” train 的向量；打标签自然准不住，最后就只有 ~50% 的随机水平。

# freeze_bert=True 下的 BertClassifier: 虽然把 BERT 的参数都 requires_grad=False，避免梯度计算和反传，
# 但在每个 batch 里，还是要把输入的 input_ids, attention_mask 喂给 BERT，执行一次完整的前向!
# 即使是在 with torch.no_grad() 里，这部分计算开销跟正常 fine‑tune 是一样的 —— 只是多了 “不算梯度” 的开销优化，但依然要过所有层。
# 因此其实二者结果是一样的，但是这种就花费很多时间！

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