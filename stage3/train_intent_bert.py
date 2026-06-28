import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# -------------------------
# 1. LOAD CSV DATASET
# -------------------------
df = pd.read_csv("stage3/bert_train_dataset.csv")

texts = df["text"].tolist()
labels_raw = df["intent"].tolist()

label_list = sorted(list(set(labels_raw)))
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

# -------------------------
# 2. MODEL
# -------------------------
model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    label2id=label2id,
    id2label=id2label
)

# -------------------------
# 3. DATASET CLASS
# -------------------------
class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True
        )
        self.labels = [label2id[l] for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

dataset = IntentDataset(texts, labels_raw)

# -------------------------
# 4. TRAINING SETTINGS
# -------------------------
training_args = TrainingArguments(
    output_dir="stage3/intent_model",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=5,
    save_strategy="epoch"
)

# -------------------------
# 5. TRAINER
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# -------------------------
# 6. TRAIN
# -------------------------
trainer.train()

# -------------------------
# 7. SAVE
# -------------------------
model.save_pretrained("stage3/intent_model")
tokenizer.save_pretrained("stage3/intent_model")

print("Training complete. Model saved!")