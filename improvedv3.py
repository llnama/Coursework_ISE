import pandas as pd
import numpy as np
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords

# nltk.download("stopwords")


def remove_html(text):
    return re.sub(r"<.*?>", "", text)


def remove_emoji(text):
    pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f1e0-\U0001f1ff"
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]",
        flags=re.UNICODE,
    )
    return pattern.sub(r"", text)


stop_words = stopwords.words("english") + ["..."]


def remove_stopwords(text):
    return " ".join(word for word in str(text).split() if word not in stop_words)


def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", text)
    text = re.sub(r"\'s", " 's", text)
    text = re.sub(r"\'ve", " 've", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r'"', "", text)
    return text.strip().lower()


# ===== CONFIG =====
REPEAT = 10
project = "caffe"
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_folder = os.path.join(script_dir, "datasets")
data_path = os.path.join(dataset_folder, f"{project}.csv")
output_path = os.path.join(script_dir, f"{project}_results_v3.csv")

# ===== LOAD & CLEAN DATA =====
df = pd.read_csv(data_path).fillna("")

df["Title+Body"] = df.apply(
    lambda row: row["Title"] + ". " + row["Body"]
    if pd.notna(row["Body"])
    else row["Title"],
    axis=1,
)
df = df.rename(columns={"Unnamed: 0": "id", "Labels": "label", "Title+Body": "text"})

df["text"] = df["text"].apply(remove_html)
df["text"] = df["text"].apply(remove_emoji)
df["text"] = df["text"].apply(remove_stopwords)
df["text"] = df["text"].apply(clean_str)

df = df[df["label"].notna() & (df["label"].str.strip() != "")]
df = df.drop_duplicates(subset=["text", "label"])

# ===== DROP RARE CLASSES (<3 samples) =====
label_counts = df["label"].value_counts()
df = df[df["label"].isin(label_counts[label_counts >= 3].index)]
df = df.reset_index(drop=True)

# ===== HYPERPARAM GRID =====
params = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2"],
    "solver": ["liblinear"],
}

accuracies, precisions, recalls, f1s, aucs = [], [], [], [], []

for seed in range(REPEAT):
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=seed
    )

    X_train = train_df["text"]
    X_test = test_df["text"]
    y_train = train_df["label"]
    y_test = test_df["label"]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = GridSearchCV(
        LogisticRegression(class_weight="balanced", max_iter=1000),
        params,
        cv=StratifiedKFold(n_splits=3),
        scoring="accuracy",
    )
    clf.fit(X_train_vec, y_train)

    best = clf.best_estimator_
    y_pred = best.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    try:
        auc_score = roc_auc_score(
            pd.get_dummies(y_test),
            pd.get_dummies(y_pred),
            average="macro",
            multi_class="ovr",
        )
    except:
        auc_score = 0.0

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)
    aucs.append(auc_score)

# ===== RESULTS =====
print("=== Logistic Regression + TF-IDF + Class Weighting (Labels) ===")
print(f"Number of repeats:     {REPEAT}")
print(f"Average Accuracy:      {np.mean(accuracies):.4f}")
print(f"Average Precision:     {np.mean(precisions):.4f}")
print(f"Average Recall:        {np.mean(recalls):.4f}")
print(f"Average F1 score:      {np.mean(f1s):.4f}")
print(f"Average AUC:           {np.mean(aucs):.4f}")

log_df = pd.DataFrame(
    {
        "repeated_times": [REPEAT],
        "Accuracy": [np.mean(accuracies)],
        "Precision": [np.mean(precisions)],
        "Recall": [np.mean(recalls)],
        "F1": [np.mean(f1s)],
        "AUC": [np.mean(aucs)],
    }
)
log_df.to_csv(output_path, index=False, mode="w", header=True)
print(f"\nResults have been saved to: {output_path}")

# ===== CUSTOM PREDICTIONS =====
custom_texts = [
    "Error: Cannot compile on macOS Sonoma. The build system fails due to Apple Silicon compatibility problems.",
    "Issue: This library is not compatible with Python 3.12. The installation fails with version mismatch errors.",
    "Question: Does this framework support GPU acceleration on Windows? I only see CPU usage during training.",
]

print("\n=== Sample Predictions ===")
for i, text in enumerate(custom_texts, 1):
    clean = clean_str(remove_stopwords(remove_emoji(remove_html(text))))
    vec = vectorizer.transform([clean])
    pred_label = best.predict(vec)[0]
    print(f"\nSample #{i}")
    print(f"Input: {text}")
    print(f"Predicted Label: {pred_label}")
