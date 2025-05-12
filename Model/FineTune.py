# finetune.py

import os

from torch.utils.data import DataLoader
import torch.optim as optim
from datasets import Dataset
from DataHandler.CustomizedDataLoader import load_split_raw, tokenize_batch
from transformers import AutoTokenizer
from DataHandler.Preprocess import save_pickle
from Pipeline import LoRABertClassifier, run_classification

def main():
    model_name = "bert-base-multilingual-uncased"
    out_dir = "../results/finetune"
    batch_size = 32

    # Load mixed split
    (train_texts, train_labels, train_sources), (dev_texts, dev_labels, dev_sources), (test_texts, test_labels,test_sources) = load_split_raw()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ========Start Construct HuggingFace Dataset========
    ds_train = Dataset.from_dict({"text": train_texts, "labels": train_labels, "source": train_sources})
    ds_dev   = Dataset.from_dict({"text": dev_texts,   "labels": dev_labels, "source": dev_sources})
    ds_test  = Dataset.from_dict({"text": test_texts,  "labels": test_labels, "source": test_sources})

    # Tokenize (batched)
    ds_train = ds_train.map(lambda b: tokenize_batch(b["text"], tokenizer), batched=True)
    ds_dev   = ds_dev.map(lambda b: tokenize_batch(b["text"], tokenizer), batched=True)
    ds_test  = ds_test.map(lambda b: tokenize_batch(b["text"], tokenizer), batched=True)

    # Since we have to input "source", so we should set output_all_columns=True while removing all other columns
    ds_train = ds_train.remove_columns(column_names=["text", "token_type_ids"])
    ds_dev = ds_dev.remove_columns(column_names=["text", "token_type_ids"])
    ds_test = ds_test.remove_columns(column_names=["text", "token_type_ids"])

    # Set format
    ds_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"], output_all_columns=True)
    ds_dev.set_format("torch", columns=["input_ids", "attention_mask", "labels"],output_all_columns=True)
    ds_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"], output_all_columns=True)

    # DataLoader
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(ds_dev, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    # ========End Construct HuggingFace Dataset========

    # ========Start Fine-tuning========
    # Use same model and pre-trained weights, run 5 epochs fine-tuning
    # finetune_model = BertClassifier(model_name=model_name, unfreeze_last_n=2)
    # # Load previous trained classifier parameters?
    # finetune_metrics = run_classification(
    #     model=finetune_model,
    #     train_loader=train_loader,
    #     dev_loader=dev_loader,
    #     test_loader=test_loader,
    #     optimizer=optim.AdamW(finetune_model.parameters(), lr=2e-5, weight_decay=0.01),
    #     epoch=100,
    #     name="Mixed_BERT_FineTune"
    # )
    # save_pickle(finetune_metrics, os.path.join(out_dir, "finetune_metrics.pkl"))
    # ========End Fine-tuning========

    # # ========Start LoRA Fine-tuning========
    lora_model = LoRABertClassifier(model_name=model_name, lora_r=16, lora_alpha=32)
    lora_metrics = run_classification(
        model=lora_model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        optimizer=optim.AdamW(lora_model.parameters(), lr=1e-4, weight_decay=0.01),
        epoch=100,
        name="Mixed_BERT_LoRA"
    )
    save_pickle(lora_metrics, os.path.join(out_dir, "lora_metrics.pkl"))
    # # ========End LoRA Fine-tuning========

    # ========Start Adapter Fine-tuning========
    # adapter_model = AdapterBertClassifier(model_name=model_name, adapter_name="sentiment")
    # adapter_metrics = run_classification(
    #     model=adapter_model,
    #     train_loader=train_loader,
    #     dev_loader=dev_loader,
    #     test_loader=test_loader,
    #     optimizer=optim.AdamW(adapter_model.parameters(), lr=2e-5, weight_decay=0.01),
    #     epoch=100,
    #     name="Mixed_BERT_adapter"
    # )
    # save_pickle(adapter_metrics, os.path.join(out_dir, "adapter_metrics.pkl"))
    # ========End LoRA Fine-tuning========


if __name__ == "__main__":
    main()
