import os
import re
import jieba
import pickle
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.model_selection import train_test_split

# Display detailed information (non-null counts) for up to 3,000,000 rows when using df.info().
pd.options.display.max_info_rows = 3000000

VERBOSE = True

# Load English stopwords
nltk.download("stopwords")
en_stopwords = set(stopwords.words("english"))

# Load Chinese stopwords
zh_stopwords_path = "zh_stopwords.txt"
with open(zh_stopwords_path, "r", encoding="utf-8") as f:
    zh_stopwords = set([line.strip() for line in f if line.strip()])

def load_imdb(path, verbose=True):
    """
    Load IMDB Dataset.

    Parameters:
    - path: raw_data path
    - verbose: print details if set to True
    """
    df = pd.read_csv(path)

    if verbose:
        print("=============================IMDB Dataset=============================")
        print("Raw IMDB Dataset:")
        df.info()
        print(df.head(3))

    imdb_df = df.copy()
    imdb_df["label"] = imdb_df["sentiment"].map({"positive": 1, "negative": 0})
    # Drop rows with missing data
    imdb_df = imdb_df.dropna(subset=["review", "label"])
    imdb_df = imdb_df[["review", "label"]]
    imdb_df["source"] = "imdb"

    return imdb_df

def load_douban(path, verbose=True):
    """
    Load Douban Dataset.

    Parameters:
    - path: raw_data path
    - verbose: print details if set to True
    """
    df = pd.read_csv(path)

    if verbose:
        print("=============================Douban Dataset=============================")
        print("Raw Douban Dataset:")
        df.info()
        print(df.head(3))

    douban_df = df.copy()
    douban_df = douban_df[df["RATING"].isin([1, 2, 3, 4, 5])]
    # Remove rows with empty comment content
    douban_df = douban_df.dropna(subset=["CONTENT"])

    def rating_to_label(r):
        if r >= 4:
            return 1
        elif r <= 2:
            return 0
        # if r = 3, mark it as neutral and remove it
        else:
            return None

    douban_df["label"] = douban_df["RATING"].apply(rating_to_label)
    # remove NaNs
    douban_df = douban_df[douban_df["label"].notnull()]
    douban_df["label"] = douban_df["label"].astype(int)
    douban_df = douban_df[["CONTENT", "label"]].rename(columns={"CONTENT": "review"})
    douban_df["source"] = "douban"

    return douban_df

def split_and_save_dataset(df, save_dir, prefix="", random_state=42, verbose=True):
    """
    Split a DataFrame into train/dev/test and save as CSV files.

    Parameters:
    - df: pd.DataFrame with 'review' and 'label' columns
    - save_dir: folder to save CSV files
    - prefix: file prefix, like 'imdb_' or 'douban_'
    - random_state: for reproducibility
    """
    os.makedirs(save_dir, exist_ok=True)

    # 70% train, 10% dev, 20% test
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=random_state, stratify=df["label"])
    dev_df, test_df = train_test_split(temp_df, test_size=2/3, random_state=random_state, stratify=temp_df["label"])

    # Save to CSV
    train_df.to_csv(os.path.join(save_dir, f"{prefix}train.csv"), index=False)
    dev_df.to_csv(os.path.join(save_dir, f"{prefix}dev.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, f"{prefix}test.csv"), index=False)

    if verbose:
        print("=============================raw_data Split=============================")
        print(f"{prefix} raw_data split & saved:")
        print(f"Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

def clean_dataset(df, language="en", deep = True, verbose=True):
    df = df.copy()
    df["review"] = df["review"].astype(str)

    if deep:
        df["review"] = df["review"].apply(deep_clean, args=(language,))
    else:
        df["review"] = df["review"].apply(basic_clean)
    df = df.dropna(subset=["review"])
    df = df[df["review"].str.strip().astype(bool)]
    df = df.reset_index(drop=True)

    if verbose:
        print(f"\nCleaned Dataset (Deep CLean: {deep}):")
        df.info()
        print(df.head(3))

    return df

def deep_clean(text, language="en"):
    """
    Clean input text for NLP tasks.

    Parameters:
    - text: Raw text
    - language: 'en' for English or 'zh' for Chinese
    - remove_stopwords: Whether to remove stopwords
    """
    text = basic_clean(text)

    # Remove emojis and other non-alphanumeric unicode characters
    if language == "zh":
        # For Chinese: keep only Chinese characters, letters, numbers.
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)
    else:
        # For English: remove all punctuation and special characters.
        text = re.sub(r"[^\w\s]", " ", text)

    # Normalize whitespaces (convert multiple spaces to one, remove leading/trailing spaces)
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize and remove stopwords if needed
    if language == "zh":
        # For Chinese: Use jieba to tokenize, remove words in zh_stopwords.
        tokens = [w for w in jieba.cut(text) if w not in zh_stopwords]
    else:
        # For English: split by spaces, lowercase, and remove English stopwords.
        tokens = [w.lower() for w in text.split() if w.lower() not in en_stopwords]
    return " ".join(tokens)

def basic_clean(text):
    if not isinstance(text, str):
        return ""
    # Only remove HTML/URL, but keep punctuations
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Normalize whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def save_pickle(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def main():
    print("Preprocessing raw data and splitting...")

    # Load and clean raw data
    imdb_df = load_imdb("raw_data/IMDB_Dataset.csv", verbose=VERBOSE)
    douban_df = load_douban("raw_data/Douban_Dataset.csv", verbose=VERBOSE)

    # Balance Dataset
    pos = douban_df[douban_df["label"] == 1]
    neg = douban_df[douban_df["label"] == 0]
    n = min(len(pos), len(neg))

    douban_df_bal = pd.concat([
        pos.sample(n, random_state=42),
        neg.sample(n, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    douban_df = douban_df_bal

    # Basic Clean (for bert): 只去 HTML/URL、合并空白，保留标点/大小写/停用词
    imdb_df = clean_dataset(imdb_df, language="en", deep = False, verbose=VERBOSE)
    douban_df = clean_dataset(douban_df, language="zh", deep=False, verbose=VERBOSE)

    # Downsampling to keep dataset balance
    douban_df, _ = train_test_split(douban_df, train_size=len(imdb_df), stratify=douban_df["label"], random_state=42)

    # Split and save csv (Basic Cleaned raw text, which can be fed into bert)
    split_and_save_dataset(imdb_df, "raw_data/splits", prefix="imdb_", random_state=42, verbose=True)
    split_and_save_dataset(douban_df, "raw_data/splits", prefix="douban_", random_state=42, verbose=True)

    # Load Splits
    imdb_train = pd.read_csv("raw_data/splits/imdb_train.csv")
    imdb_dev = pd.read_csv("raw_data/splits/imdb_dev.csv")
    imdb_test = pd.read_csv("raw_data/splits/imdb_test.csv")
    douban_train = pd.read_csv("raw_data/splits/douban_train.csv")
    douban_dev = pd.read_csv("raw_data/splits/douban_dev.csv")
    douban_test = pd.read_csv("raw_data/splits/douban_test.csv")

    # Deep Clean: Remove Stopwords
    imdb_train = clean_dataset(imdb_train, language="en", verbose=VERBOSE)
    imdb_dev = clean_dataset(imdb_dev, language="en", verbose=VERBOSE)
    imdb_test = clean_dataset(imdb_test, language="en", verbose=VERBOSE)
    douban_train = clean_dataset(douban_train, language="zh", verbose=VERBOSE)
    douban_dev = clean_dataset(douban_dev, language="zh", verbose=VERBOSE)
    douban_test = clean_dataset(douban_test, language="zh", verbose=VERBOSE)

    # Save labels
    imdb_train_labels = imdb_train["label"].tolist()
    imdb_dev_labels = imdb_dev["label"].tolist()
    imdb_test_labels = imdb_test["label"].tolist()
    douban_train_labels = douban_train["label"].tolist()
    douban_dev_labels = douban_dev["label"].tolist()
    douban_test_labels = douban_test["label"].tolist()

    save_pickle(imdb_train_labels, "raw_data/labels/imdb_train_labels.pkl")
    save_pickle(imdb_dev_labels, "raw_data/labels/imdb_dev_labels.pkl")
    save_pickle(imdb_test_labels, "raw_data/labels/imdb_test_labels.pkl")
    save_pickle(douban_train_labels, "raw_data/labels/douban_train_labels.pkl")
    save_pickle(douban_dev_labels, "raw_data/labels/douban_dev_labels.pkl")
    save_pickle(douban_test_labels, "raw_data/labels/douban_test_labels.pkl")

    # Tokenized at sentence level
    imdb_sentences_train = [text.split() for text in imdb_train["review"]]
    imdb_sentences_dev = [text.split() for text in imdb_dev["review"]]
    imdb_sentences_test = [text.split() for text in imdb_test["review"]]

    douban_sentences_train = [text.split() for text in douban_train["review"]]
    douban_sentences_dev = [text.split() for text in douban_dev["review"]]
    douban_sentences_test = [text.split() for text in douban_test["review"]]

    # Tokenized at word level from training dataset for the Zipf analysis
    imdb_tokens_train = [token for sent in imdb_sentences_train for token in sent]
    douban_tokens_train = [token for sent in douban_sentences_train for token in sent]

    # Save tokens
    save_pickle(imdb_tokens_train, "tokens/zipf/imdb_tokens_train.pkl")
    save_pickle(douban_tokens_train, "tokens/zipf/douban_tokens_train.pkl")

    save_pickle(imdb_sentences_train, "tokens/sentences/imdb_sentences_train.pkl")
    save_pickle(imdb_sentences_dev, "tokens/sentences/imdb_sentences_dev.pkl")
    save_pickle(imdb_sentences_test, "tokens/sentences/imdb_sentences_test.pkl")
    save_pickle(douban_sentences_train, "tokens/sentences/douban_sentences_train.pkl")
    save_pickle(douban_sentences_dev, "tokens/sentences/douban_sentences_dev.pkl")
    save_pickle(douban_sentences_test, "tokens/sentences/douban_sentences_test.pkl")

if __name__ == "__main__":
    main()