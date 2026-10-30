
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# 1. Load dataset
df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

X = df["message"]
y = df["label"]

# 2. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Define models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
}

results = []

# 4. Train and evaluate both models
for name, classifier in models.items():

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, pos_label="spam"
    )
    recall = recall_score(
        y_test, predictions, pos_label="spam"
    )
    f1 = f1_score(
        y_test, predictions, pos_label="spam"
    )

    print(f"\n--- {name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    print("Confusion matrix:")
    print(confusion_matrix(
        y_test, predictions, labels=["ham", "spam"]
    ))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    # Save each trained model separately
    filename = (
        "naive_bayes.joblib"
        if name == "Naive Bayes"
        else "logistic_regression.joblib"
    )
    joblib.dump(pipeline, filename)

# 5. Display comparison
results_df = pd.DataFrame(results)

print("\n--- MODEL COMPARISON ---")
print(results_df.to_string(index=False))

results_df.to_csv("model_results.csv", index=False)

print("\nResults saved to model_results.csv")
