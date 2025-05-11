import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import os

def plot_zipf(tokens1, tokens2, dataset_name1, dataset_name2, top_k=50000):
    """
    Plot Zipf's Law: frequency vs rank for the dataset.

    Parameters:
    - tokens1, tokens2: list of all tokens in each dataset (flattened)
    - dataset_name1, dataset_name2: labels for the plot
    - top_k: how many most frequent tokens to plot
    """
    def get_freqs(tokens):
        counter = Counter(tokens)
        most_common = counter.most_common(top_k)
        freqs = np.array([freq for _, freq in most_common])
        ranks = np.arange(1, len(freqs) + 1)
        return ranks, freqs

    ranks1, freqs1 = get_freqs(tokens1)
    ranks2, freqs2 = get_freqs(tokens2)

    ideal_zipf = freqs1[0] / ranks1

    # Plot on log-log scale
    plt.figure(figsize=(9, 6))
    plt.loglog(ranks1, freqs1, label=dataset_name1, color="blue")
    plt.loglog(ranks2, freqs2, label=dataset_name2, color="green")
    plt.loglog(ranks1, ideal_zipf, label="Ideal Zipf", linestyle="--", color="gray")

    plt.xlabel("Rank of token (log scale)")
    plt.ylabel("Frequency of token (log scale)")
    plt.title("Zipf's Law Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/zipf_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.show()

def print_similar_words(model, word, title=None, topn=10):
    print(f"\n=== {title or 'Similar words'} for '{word}' ===")
    try:
        results = model.wv.most_similar(word, topn=topn)
        for w, score in results:
            print(f"{w:15s} | similarity: {score:.4f}")
    except KeyError:
        print(f"'{word}' not found in vocabulary.")


def plot_training_curves(train_losses, val_accuracies, epochs, name="IMDB", save_dir="results"):
    epochs_range = range(1, epochs + 1)

    # Training Loss
    plt.figure()
    plt.plot(epochs_range, train_losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title(f"Simple Classifier Training Loss on {name}")
    plt.grid(True)
    save_loss_path = os.path.join(save_dir, f"{name}_train_loss.png")
    plt.tight_layout()
    plt.savefig(save_loss_path)
    print(f"[Saved] Training loss curve → {save_loss_path}")
    plt.show()
    plt.close()

    # Validation Accuracy
    plt.figure()
    plt.plot(epochs_range, [acc * 100 for acc in val_accuracies], marker='o', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title(f"Simple Classifier Validation Accuracy on {name}")
    plt.grid(True)
    save_acc_path = os.path.join(save_dir, f"{name}_val_accuracy.png")
    plt.tight_layout()
    plt.savefig(save_acc_path)
    print(f"[Saved] Validation accuracy curve → {save_acc_path}")
    plt.show()
    plt.close()