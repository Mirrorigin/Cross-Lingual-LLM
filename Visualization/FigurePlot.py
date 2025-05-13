import os
from Utils import plot_metrics
import pickle

BASELINE = False
FINE_TUNING = False

if BASELINE:
    # Load the saved baseline metrics file
    with open("../results/baseline_all_metrics.pkl", "rb") as f:
        all_metrics = pickle.load(f)

    # Plot each metric
    plot_metrics(all_metrics, "train_losses", "Training Loss")
    plot_metrics(all_metrics, "val_accuracies", "Validation Accuracy")
    plot_metrics(all_metrics, "val_f1s", "Validation F1 Score")

if FINE_TUNING:
    # Define the directory containing all metric files
    out_dir = "../results/finetune"
    expected_files = {
        "Standard Fine-tuning": "finetune_metrics.pkl",
        "LoRA": "lora_metrics.pkl",
        "Adapter": "adapter_metrics.pkl"
    }

    # Load each file if it exists
    all_metrics = {}
    for model_name, filename in expected_files.items():
        path = os.path.join(out_dir, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                all_metrics[model_name] = pickle.load(f)
        else:
            print(f"Warning: {path} not found.")

    # Plot training loss comparison
    plot_metrics(all_metrics, "train_losses", "Training Loss", max_epoch=30)

