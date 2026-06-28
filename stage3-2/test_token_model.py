import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import classification_report, accuracy_score

model_path = "stage3-2/token_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()

id2label = model.config.id2label

df = pd.read_csv("stage3-2/bert_test_dataset.csv")

all_true = []
all_pred = []

def predict(tokens):
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    preds = torch.argmax(logits, dim=2)[0].tolist()
    word_ids = inputs.word_ids()

    pred_labels = []

    for word_id, pred in zip(word_ids, preds):
        if word_id is None:
            continue
        pred_labels.append(id2label[pred])

    return pred_labels

# ------------------------
# EVALUATION LOOP
# ------------------------
for _, row in df.iterrows():

    tokens = row["tokens"].split()
    true_labels = row["labels"].split()

    pred_labels = predict(tokens)

    # align length safety
    min_len = min(len(true_labels), len(pred_labels))

    all_true.extend(true_labels[:min_len])
    all_pred.extend(pred_labels[:min_len])

# ------------------------
# METRICS
# ------------------------
print("\n====================")
print("TOKEN MODEL RESULTS")
print("====================")

print("\nACCURACY:", accuracy_score(all_true, all_pred))

print("\nCLASSIFICATION REPORT:")
print(classification_report(all_true, all_pred))