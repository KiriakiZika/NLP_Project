import joblib
import numpy as np

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("stage2/intent_model.pkl")

# -------------------------
# PREDICTION FUNCTION
# -------------------------
def predict_intent(text, threshold=0.50):
    probs = model.predict_proba([text])[0]
    classes = model.classes_

    best_idx = np.argmax(probs)
    best_prob = probs[best_idx]
    best_label = classes[best_idx]

    if best_prob < threshold:
        return "UNKNOWN", best_prob

    return best_label, best_prob

# -------------------------
# INTERACTIVE LOOP
# -------------------------
print("\n🎯 Intent Classifier Ready!")
print("Type a command (or 'exit' to quit)\n")

while True:
    user_input = input(">> ")

    if user_input.lower() == "exit":
        break

    intent, conf = predict_intent(user_input)

    print(f"Predicted Intent: {intent}")
    print(f"Confidence: {conf:.2f}")
    print("-" * 40)