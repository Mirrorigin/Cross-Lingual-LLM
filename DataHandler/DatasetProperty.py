from pathlib import Path
import pandas as pd

# ==================== Check Raw Data ====================
# Load data, convert all empty string into NaN
df = pd.read_csv("../raw_data/Douban_Dataset.csv", na_values=["", " ", "null", "NA", "nan"])

print("Original Shape:", df.shape)
print("RATING Uniques", df["RATING"].unique())

# Rating Conversion (Same as Preprocess)
def rating_to_label(r):
    if r >= 4:
        return 1
    elif r <= 2:
        return 0
    # if r = 3, mark it as neutral and remove it
    else:
        return None

df["label"] = df["RATING"].apply(rating_to_label)
df_bin = df.dropna(subset=["label"]).copy()

label_counts = df_bin["label"].value_counts().sort_index()
label_ratio  = (label_counts / len(df_bin)).round(4)

summary = pd.concat([label_counts, label_ratio], axis=1)
summary.columns = ["count", "ratio"]

print("\nData Shape", df_bin.shape)
print(summary)

max_min = label_counts.max() / label_counts.min()
print(f"\nmax/min = {max_min:.2f}  (≤1.5 can be regarded as balanced)\n")

# ==================== Check Split Data ====================

dataset_name = "douban"

# Split data path
split_paths = {
    "train": Path(f"raw_data/splits/{dataset_name}_train.csv"),
    "dev":   Path(f"raw_data/splits/{dataset_name}_dev.csv"),
    "test":  Path(f"raw_data/splits/{dataset_name}_test.csv"),
}

rows = []
for split, path in split_paths.items():
    df = pd.read_csv(path)
    counts = df["label"].value_counts().sort_index()      # 0 / 1 counts
    total  = counts.sum()
    neg, pos = counts.get(0, 0), counts.get(1, 0)
    max_min = round(max(neg, pos) / max(1, min(neg, pos)), 2)  # avoid /0
    rows.append({
        "split": split,
        "neg":   neg,
        "pos":   pos,
        "neg_ratio": round(neg / total, 4),
        "pos_ratio": round(pos / total, 4),
        "max/min": max_min,
    })

summary = pd.DataFrame(rows).set_index("split")
print(summary)
