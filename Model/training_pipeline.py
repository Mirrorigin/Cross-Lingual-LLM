import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score
from DataHandler.data_loader import *
from Visualization.plotting_funcs import *
from transformers import AutoModel, AutoConfig, BertModel
from adapters import AutoAdapterModel, AdapterConfig
from peft import get_peft_model, LoraConfig, TaskType

# Define two models: SimpleClassifier for baseline, BertClassifier for fine-tuning
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, dropout_prob=0.1):
        super().__init__()
        # Default dropout_prob=0.1
        self.head = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(input_dim, 2)
        )

    def forward(self, x):
        # x: [batch_size, input_dim]
        return self.head(x)  # [batch_size, 2]

class BertClassifier(nn.Module):
    def __init__(self, model_name="bert-base-multilingual-uncased", dropout_prob=0.1, unfreeze_last_n=0):
        super().__init__()
        self.unfreeze_last_n = unfreeze_last_n
        # Only load backbone
        cfg = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=cfg)
        hidden_size = cfg.hidden_size
        # Use SimpleClassifier, which includes Dropout plus Linear(input_dim)
        self.classifier = SimpleClassifier(input_dim=hidden_size, dropout_prob=dropout_prob)

        # Freeze the entire bert first
        for p in self.bert.parameters():
            p.requires_grad = False
        if unfreeze_last_n > 0:
            total_layers = len(self.bert.encoder.layer)
            for layer in self.bert.encoder.layer[total_layers - unfreeze_last_n:]:
                for p in layer.parameters():
                    p.requires_grad = True

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {trainable_params} || all params: {total_params} || trainable%: {(trainable_params / total_params) * 100:.4f}%")

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls = output.last_hidden_state[:, 0]      # Get [CLS] Vectors, [B, hidden_size]
        logits = self.classifier(cls)          # (B,2)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else logits

class AdapterBertClassifier(nn.Module):
    def __init__(self, model_name="bert-base-multilingual-uncased", adapter_name="sentiment", reduction=16):
        super().__init__()
        # Load Adapter-support model
        self.bert = AutoAdapterModel.from_pretrained(model_name)
        # Add Pfeiffer Adapter
        config = AdapterConfig.load("pfeiffer", reduction_factor=reduction)
        self.bert.add_adapter(adapter_name, config=config)
        self.bert.train_adapter(adapter_name)

        hidden_size = self.bert.config.hidden_size
        self.classifier = SimpleClassifier(hidden_size, dropout_prob=0.1)

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {trainable_params} || all params: {total_params} || trainable%: {(trainable_params / total_params) * 100:.4f}%")

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        cls = output.last_hidden_state[:, 0]
        logits = self.classifier(cls)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        return logits

# Core idea of LoRA: insert low-rank adapters into the weight matrices of each layer in the Transformer.
# During each forward pass, the entire backbone must still be run in order to incorporate the effects of these adapters (i.e., the new parameters) into the output!
# Even if the original weights are completely frozen and not updated, the full series of matrix multiplications and additions still has to be executed.
# Only in this way can the fine-tuned features learned by the adapter be reflected in the CLS vector. Otherwise, the result would just be the same as the original BERT embedding, without any fine-tuning effect.
class LoRABertClassifier(nn.Module):
    def __init__(self, model_name: str = "bert-base-multilingual-uncased", dropout_prob = 0.1,
                 lora_r = 16, lora_alpha = 32, lora_dropout = 0.0):
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_name)
        backbone = BertModel.from_pretrained(model_name, config=cfg)

        # Construct LoRA config, only inject adapter
        lora_cfg = LoraConfig(
            # Use FEATURE_EXTRACTION instead of SEQ_CLS
            # so PEFT won’t expect 'labels' in its forward() and we can handle loss externally (use our SimpleClassifier)
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value", "dense"],
            lora_dropout=lora_dropout
        )
        self.bert = get_peft_model(backbone, lora_cfg)

        # Freeze weights, only train LoRA
        for name, param in self.bert.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

        # Use SimpleClassifier, which includes Dropout plus Linear(input_dim)
        hidden_size = cfg.hidden_size
        self.classifier = SimpleClassifier(hidden_size, dropout_prob)

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"trainable params: {trainable_params} || all params: {total_params} || trainable%: {(trainable_params / total_params) * 100:.4f}%")

        # self.bert.print_trainable_parameters()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls_vec = output.last_hidden_state[:, 0]
        logits = self.classifier(cls_vec)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        return logits

def train_model(model, dataloader, optimizer, criterion, device, *, timeit=False):
    model.train()

    stream = torch.cuda.current_stream(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record(stream)

    total_loss = 0
    for batch in dataloader:
        # Batch might be (x, y) or dict
        inputs = {k: v.to(device) for k, v in batch.items() if k not in ("labels", "source")}
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        out = model(**inputs)
        logits = out["logits"] if isinstance(out, dict) else out

        loss = criterion(logits, labels)
        loss.backward()
        # Gradient clipping: keeping the direction of the gradient unchanged, but reducing its magnitude to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    end_event.record(stream)
    torch.cuda.synchronize()
    elapsed_s = start_event.elapsed_time(end_event) / 1000.0   # seconds
    avg_loss = total_loss / len(dataloader)
    return (avg_loss, elapsed_s) if timeit else (avg_loss, None)

def evaluate_metrics(model, dataloader, device, group_key="source"):
    """Return overall_acc, overall_f1, per_group_dict"""
    model.eval()
    all_true, all_pred = [], []
    g_true = defaultdict(list)
    g_pred = defaultdict(list)

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ("labels", group_key)}
            labels = batch["labels"].to(device)
            out = model(**inputs)
            logits = out["logits"] if isinstance(out, dict) else out

            preds = logits.argmax(dim=1)

            # ——Global —— #
            all_true.extend(labels.cpu())
            all_pred.extend(preds.cpu())

            # —— Separate —— #
            for g, y, p in zip(batch[group_key], labels.cpu(), preds.cpu()):
                g_true[g].append(y)
                g_pred[g].append(p)

    overall_acc = accuracy_score(all_true, all_pred)
    overall_f1  = f1_score(all_true, all_pred, average="macro")

    per_group = {}
    for g in g_true:
        per_group[g] = {
            "acc": accuracy_score(g_true[g], g_pred[g]),
            "f1":  f1_score(g_true[g],  g_pred[g], average="macro"),
            "n":   len(g_true[g]),
        }
    return overall_acc, overall_f1, per_group

def evaluate_and_print_errors(model, dataloader, device, threshold=0.8, group_key="source"):
    model.eval()
    batch_size = dataloader.batch_size

    all_true, all_pred = [], []
    g_true = defaultdict(list)
    g_pred = defaultdict(list)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ("labels", group_key)}
            labels = batch["labels"].to(device)
            out = model(**inputs)
            logits = out["logits"] if isinstance(out, dict) else out

            preds  = logits.argmax(dim=1)
            probs = F.softmax(logits, dim=1)
            confs  = probs.max(dim=1).values

            all_true.extend(labels.cpu())
            all_pred.extend(preds.cpu())

            for g, y, p in zip(batch[group_key], labels.cpu(), preds.cpu()):
                g_true[g].append(y)
                g_pred[g].append(p)

            # Dataloader for test: shuffle = False, so directly use batch_id!
            for i, (y_true, y_pred, conf) in enumerate(zip(
                    labels.cpu().tolist(),
                    preds.cpu().tolist(),
                    confs.cpu().tolist())):
                if (y_true != y_pred) and (conf > threshold):
                    sample_id = batch_idx * batch_size + i
                    print(f">>> Outrageous’ examples detected! Sample ID: {sample_id} \t True: {y_true}, Pred: {y_pred}, Conf: {conf:.4f}")

    overall_acc = accuracy_score(all_true, all_pred)
    overall_f1  = f1_score(all_true, all_pred, average="macro")

    per_group = {}
    for g in g_true:
        per_group[g] = {
            "acc": accuracy_score(g_true[g], g_pred[g]),
            "f1":  f1_score(g_true[g],  g_pred[g], average="macro"),
            "n":   len(g_true[g]),
        }

    return overall_acc, overall_f1, per_group

def run_classification(model, train_loader, dev_loader, test_loader=None,
                       epoch = None, optimizer=None, criterion=None, name="IMDB"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = criterion or nn.CrossEntropyLoss()
    epochs = epoch or 5

    # # EarlyStopping
    # patience = 5
    # no_improve_counter = 0
    best_f1 = 0.0
    best_ckpt = os.path.join("../results", f"{name}_best.pt")

    train_losses = []
    val_accuracies = []
    val_f1s = []
    print(f"\n=== Training Classifier on {name} with {device}===")
    for epoch in range(epochs):
        train_loss, gpu_time = train_model(model, train_loader, optimizer, criterion, device, timeit=True)
        val_acc, val_f1, val_grp = evaluate_metrics(model, dev_loader, device)
        imdb_acc = val_grp.get("imdb", {}).get("acc", 0)
        dou_acc = val_grp.get("douban", {}).get("acc", 0)

        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)

        # If F1 better, then save the model
        if val_f1 > best_f1:
            best_f1 = val_f1
            # no_improve_counter = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"New best F1: {best_f1:.4f}, saved checkpoint to {best_ckpt}")
        # else:
        #     no_improve_counter += 1
        #     if no_improve_counter >= patience:
        #         print(f"[Early Stop] No improvement after {patience} epochs.")
        #         break

        if (epochs > 10 and (epoch == 0 or (epoch+1) % 5  == 0)) or (epochs <= 10):
            print(f"Epoch {epoch + 1}/{epochs} "
                  f"- Train Loss: {train_loss:.4f} "
                  f"- Val Acc: {val_acc * 100:.2f}% "
                  f"- Val F1 (macro): {val_f1:.4f}"
                  f" (imdb {imdb_acc*100:.2f}%, douban {dou_acc*100:.2f}%)"
                  f" - GPU Time: {gpu_time:.3f}s"
              )

    results = {
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
        "val_f1s": val_f1s,
    }

    if test_loader is not None:
        test_acc, test_f1, test_grp = evaluate_and_print_errors(model, test_loader, device)
        print(f"[{name}] Mixed  Acc {test_acc * 100:.2f}%  F1 {test_f1:.4f}")
        for g, m in test_grp.items():
            print(f"          {g:<6} n={m['n']:5d}  Acc {m['acc'] * 100:.2f}%  F1 {m['f1']:.4f}")

        results.update({"test_accuracy": test_acc, "test_f1": test_f1, "test_breakdown": test_grp})

    plot_training_curves(train_losses, val_accuracies, epochs, name=name, save_dir="../results")
    return results
