# train.py
# Crisis Classifier — DeBERTa-v3-base, 6-class, RTX 4050 6GB

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    set_seed,
)
from sklearn.metrics import f1_score, classification_report
from collections import Counter

# ── REPRODUCIBILITY ──────────────────────────────────────
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── CONFIG ────────────────────────────────────────────────
MODEL_NAME  = "microsoft/deberta-v3-base"
MAX_LEN     = 256
BATCH_SIZE  = 8        # 6GB VRAM — do not increase
ACCUM_STEPS = 4        # effective batch size = 8 * 4 = 32
EPOCHS      = 5
LR          = 2e-5
WARMUP_FRAC = 0.1

# ── DEVICE ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── LABEL MAPPING ─────────────────────────────────────────
LABEL2ID = {
    "no_crisis":          0,
    "suicidal_ideation":  1,
    "self_directed_harm": 2,
    "sexual_violence":    3,
    "domestic_violence":  4,
    "acute_crisis":       5,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# ── DATA LOADING ──────────────────────────────────────────
def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f]

print("\nLoading data...")
train_data = load_jsonl('data/processed/crisis/train.jsonl')
val_data   = load_jsonl('data/processed/crisis/val.jsonl')
test_data  = load_jsonl('data/processed/crisis/test.jsonl')

print(f"  Train: {len(train_data)}")
print(f"  Val:   {len(val_data)}")
print(f"  Test:  {len(test_data)}")

print("\nTrain distribution:")
dist = Counter(r['label'] for r in train_data)
for label, count in sorted(dist.items(), key=lambda x: -x[1]):
    print(f"  {label:<22} {count:>5}")

# ── CLASS WEIGHTS — computed from actual counts ───────────
total = len(train_data)
class_weights = torch.tensor([
    total / (NUM_LABELS * dist.get(ID2LABEL[i], 1))
    for i in range(NUM_LABELS)
], dtype=torch.float).to(device)

print("\nClass weights:")
for i in range(NUM_LABELS):
    print(f"  {ID2LABEL[i]:<22} {class_weights[i]:.3f}")

# ── DATASET ───────────────────────────────────────────────
class CrisisDataset(Dataset):
    def __init__(self, records, tokenizer, max_len=MAX_LEN):
        self.records  = records
        self.tok      = tokenizer
        self.max_len  = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r   = self.records[idx]
        enc = self.tok(
            r['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(LABEL2ID[r['label']], dtype=torch.long)
        }

# ── TOKENIZER + MODEL ─────────────────────────────────────
print(f"\nLoading tokenizer and model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
).float().to(device)  # Fixed: Added .float() to ensure FP32

train_ds = CrisisDataset(train_data, tokenizer)
val_ds   = CrisisDataset(val_data,   tokenizer)
test_ds  = CrisisDataset(test_data,  tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# ── OPTIMIZER + SCHEDULER + LOSS ──────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_steps   = (len(train_loader) // ACCUM_STEPS) * EPOCHS
warmup_steps  = int(WARMUP_FRAC * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# ── EVAL FUNCTION ─────────────────────────────────────────
def evaluate(loader, split_name="val"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)  # Fixed: Added missing forward pass
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return macro_f1, all_preds, all_labels

# ── TRAINING LOOP ─────────────────────────────────────────
print("\nStarting training...")
os.makedirs('models/crisis_classifier/best', exist_ok=True)

best_val_f1 = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss  = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = loss_fn(outputs.logits, labels)
        loss    = loss / ACCUM_STEPS

        loss.backward()

        if (step + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * ACCUM_STEPS

    avg_loss = total_loss / len(train_loader)
    val_f1, _, _ = evaluate(val_loader, "val")

    print(f"Epoch {epoch+1}/{EPOCHS} | loss={avg_loss:.4f} | val_macro_f1={val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        model.save_pretrained('models/crisis_classifier/best')
        tokenizer.save_pretrained('models/crisis_classifier/best')
        print(f"  -> best model saved (val_macro_f1={val_f1:.4f})")

# ── FINAL TEST EVALUATION ─────────────────────────────────
print("\nLoading best model for test evaluation...")
model = AutoModelForSequenceClassification.from_pretrained(
    'models/crisis_classifier/best',  # Fixed: Load from saved path instead of MODEL_NAME
    num_labels=NUM_LABELS
).float().to(device)  # Fixed: Added .float()

test_f1, test_preds, test_labels = evaluate(test_loader, "test")

target_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
print("\nTEST SET RESULTS:")
print(classification_report(
    test_labels, test_preds,
    target_names=target_names,
    digits=3
))
print(f"Test macro F1: {test_f1:.4f}")