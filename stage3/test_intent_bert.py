from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score


# -------------------------
# 1. LOAD MODEL
# -------------------------
model_path = "stage3/intent_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.eval()

# -------------------------
# 2. LOAD TEST DATASET
# -------------------------
df = pd.read_csv("stage3/bert_test_dataset.csv")

test_texts = df["text"].tolist()
test_labels = df["intent"].tolist()

# -------------------------
# 3. PREDICTION FUNCTION
# -------------------------

def predict(text, THRESHOLD = 0.70):
    def is_garbage(text):
        if len(text.strip()) < 2:
            return True

        if len(set(text)) == 1:  # like "aaaaaa"
            return True

        return False
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    if is_garbage(text):
        return "unknown", 0.0

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits, dim=1)[0]

    confidence, pred_id = torch.max(probs, dim=0)

    confidence = confidence.item()
    pred_id = pred_id.item()

    label = model.config.id2label[pred_id]

    # optional unknown handling
    if confidence < THRESHOLD:
        label = "unknown"

    return label, confidence

# -------------------------
# 4. RUN PREDICTIONS
# -------------------------
preds = []

for text in test_texts:
    label, conf = predict(text)
    preds.append(label)

# -------------------------
# 5. ACCURACY
# -------------------------
acc = accuracy_score(test_labels, preds)
all_labels = list(set(test_labels + ["unknown"]))

print("\n====================")
print("ACCURACY:", acc)
print("====================\n")

# -------------------------
# 6. SHOW MISTAKES (VERY IMPORTANT FOR REPORT)
# -------------------------
print("MISCLASSIFICATIONS:\n")

for text, true, pred in zip(test_texts, test_labels, preds):
    if true != pred:
        print(f"TEXT : {text}")
        print(f"TRUE : {true}")
        print(f"PRED : {pred}")
        print("----------------------")

print("\n==================== HARD TEST RESULTS ====================")

print("Accuracy:", accuracy_score(test_labels, preds))

print("\nCLASSIFICATION REPORT:")
print(classification_report(test_labels, preds, labels=all_labels))

# -------------------------
# 7. QUICK MANUAL TEST MODE
# -------------------------
print("\nTYPE SENTENCES TO TEST (type 'exit' to stop)\n")

while True:
    text = input(">> ")
    if text.lower() == "q":
        break
    intent, conf = predict(text)

    print(f"Predicted Intent: {intent}")
    print(f"Confidence: {conf:.2f}")