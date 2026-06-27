import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

#Train model and save it to a file
def train_model():
    # Read file, check contents
    df = pd.read_csv("stage2/commands.csv")
    print("\nIntents and their counts:")
    print(df["intent"].value_counts())


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["intent"],
        test_size=0.2,
        random_state=42,
        stratify=df["intent"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Train model
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, "stage2/intent_model.pkl")

    return model, X_test, y_test
model, X_test, y_test = train_model()


# Use this wherever you load the model
# If condifence<threshold return UNKNOWN
def predict_intent(text, threshold=0.50):
    probs = model.predict_proba([text])[0]
    classes = model.classes_

    best_idx = np.argmax(probs)
    best_prob = probs[best_idx]
    best_label = classes[best_idx]

    if best_prob < threshold:
        return "UNKNOWN", best_prob
    else:
        return best_label, best_prob

#Test the model with some hard-coded examples
def test_model():
    test_data = [

    # =========================
    # PICK UP
    # =========================
    ("grab the thing that is red and square", "pick_up"),
    ("take up object c1 now", "pick_up"),
    ("lift whatever is in front of the crane", "pick_up"),
    ("can you secure item b2", "pick_up"),
    ("collect the blue square object", "pick_up"),
    ("yo pick that up", "pick_up"),
    ("retrieve c3 from the platform", "pick_up"),
    ("grab the red cube", "pick_up"),
    ("lift object c2", "pick_up"),
    ("take the green square", "pick_up"),

    # =========================
    # PLACE (ONLY object → object)
    # =========================
    ("move c1 on b2", "place"),
    ("move red square on blue square", "place"),
    ("move c2 on c3", "place"),
    ("move b2 on p1", "place"),
    ("move c1 on r1", "place"),
    ("move l1 on c4", "place"),
    ("stack s1 on s2", "place"),
    ("stack green triangle on yellow square", "place"),
    ("move c3 on b3", "place"),
    ("put r1 on l2", "place"),
    ("stack c4 on c5", "place"),
    ("put blue circle on green square", "place"),
    ("move c2 on r2", "place"),
    ("stack c6 on b1", "place"),
    ("put small square on large square", "place"),
    ("move c5 on p2", "place"),
    ("stack purple triangle on cyan circle", "place"),
    ("put white cube on black square", "place"),
    ("move c3 on l1", "place"),
    ("put b2 on c2", "place"),
    ("stack r2 on s1", "place"),
    ("move c1 on c2", "place"),
    ("put green square on red square", "place"),
    ("stack yellow square on c4", "place"),
    ("move orange square on b3", "place"),
    ("stack c2 on l3", "place"),
    ("put c4 on r1", "place"),
    ("move c6 on s2", "place"),
    ("put blue square on p1", "place"),
    ("move c5 on b2", "place"),

    # =========================
    # PUT_DOWN
    # =========================
    ("release whatever you are holding", "put_down"),
    ("drop the object immediately", "put_down"),
    ("set it back on the ground", "put_down"),
    ("can you put that thing down now", "put_down"),
    ("free the item from the crane", "put_down"),
    ("unload current object", "put_down"),
    ("let go of c1", "put_down"),
    ("drop it", "put_down"),
    ("put it down", "put_down"),
    ("release object", "put_down"),

    # =========================
    # SCRAMBLED / NOISY
    # =========================
    ("red square grab the", "pick_up"),
    ("b2 on stack c1 put", "place"),
    ("down put it", "put_down"),
    ("c3 drop now", "put_down"),
    ("square blue grab the", "pick_up"),

    # =========================
    # AMBIGUOUS / OUT OF DISTRIBUTION
    # =========================
    ("do the thing", "put_down"),
    ("move it there", "place"),
    ("grab something", "pick_up"),
    ("handle object properly", "put_down"),
    ("execute pickup sequence", "pick_up"),
    ("reposition asset", "place"),
    ("stack appropriately", "place"),
    ("release now", "put_down"),
    ]
    for text, true_label in test_data:
        pred_label, conf = predict_intent(text)

        print(f"{text}")
        print(f"  TRUE: {true_label}")
        print(f"  PRED: {pred_label} (confidence={conf:.2f})")
        print("-" * 60)

    y_true = []
    y_pred = []

    for text, true_label in test_data:
        pred_label, conf = predict_intent(text)

        y_true.append(true_label)
        y_pred.append(pred_label)

    from sklearn.metrics import classification_report, accuracy_score

    print("\nHARD TEST RESULTS")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))
test_model()