import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import torch

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv("stage3-2/bert_train_dataset.csv")

# -----------------------
# LABEL SET
# -----------------------
label_list = ["O", "B-OBJ", "I-OBJ", "B-DEST", "I-DEST", "B-REL"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

# -----------------------
# TOKENIZER
# -----------------------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# -----------------------
# CONVERT TEXT → TOKENS + ALIGN LABELS
# -----------------------
def preprocess(example):
    tokens = example["tokens"].split()
    labels = example["labels"].split()

    enc = tokenizer(tokens, is_split_into_words=True, truncation=True)

    word_ids = enc.word_ids()

    label_ids = []
    prev_word = None

    for word_id in word_ids:
        if word_id is None:
            label_ids.append(-100)
        elif word_id != prev_word:
            label_ids.append(label2id[labels[word_id]])
        else:
            # same word (subtoken)
            label_ids.append(label2id[labels[word_id]])
        prev_word = word_id

    enc["labels"] = label_ids
    return enc

# -----------------------
# DATASET
# -----------------------
dataset = Dataset.from_pandas(df)

#find bad rows
for i, row in df.iterrows():
    tokens = row["tokens"].split()
    labels = row["labels"].split()

    if len(tokens) != len(labels):
        print("BAD ROW:", i)
        print("TOKENS:", tokens)
        print("LABELS:", labels)
        print("------")

dataset = dataset.map(preprocess)
dataset = dataset.train_test_split(test_size=0.2)

# -----------------------
# MODEL
# -----------------------
model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# -----------------------
# METRICS
# -----------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=2)

    true_labels = []
    true_preds = []

    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] != -100:
                true_labels.append(labels[i][j])
                true_preds.append(preds[i][j])

    acc = np.mean(np.array(true_labels) == np.array(true_preds))
    return {"accuracy": acc}

# -----------------------
# TRAINING ARGS
# -----------------------
args = TrainingArguments(
    output_dir="stage3/token_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=50
)

# -----------------------
# TRAINER
# -----------------------
data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

# -----------------------
# TRAIN
# -----------------------
trainer.train()

# -----------------------
# SAVE MODEL
# -----------------------
model.save_pretrained("stage3-2/token_model")
tokenizer.save_pretrained("stage3-2/token_model")

print("Training complete. Model saved to stage3-2/token_model")