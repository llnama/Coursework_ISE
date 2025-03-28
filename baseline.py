########## 1. Import required libraries ##########

import pandas as pd
import numpy as np
import re
import math

# Text and feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer

# Evaluation and tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)

# Classifier
from sklearn.naive_bayes import GaussianNB

# Text cleaning & stopwords
import nltk

# nltk.download("stopwords")
from nltk.corpus import stopwords
import os
########## 2. Define text preprocessing methods ##########


def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r"<.*?>")
    return html.sub(r"", text)


def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"  # enclosed characters
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


# Stopwords
NLTK_stop_words_list = stopwords.words("english")
custom_stop_words_list = ["..."]  # You can customize this list as needed
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list


def remove_stopwords(text):
    """Remove stopwords from the text."""
    return " ".join(
        [word for word in str(text).split() if word not in final_stop_words_list]
    )


def clean_str(string):
    """
    Clean text by removing non-alphanumeric characters,
    and convert it to lowercase.
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


########## 3. Download & read data ##########


script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_folder = os.path.join(script_dir, "datasets")

project = "caffe"
path = os.path.join(dataset_folder, f"{project}.csv")

pd_all = pd.read_csv(path).fillna("")
pd_all = pd_all.sample(frac=1, random_state=999)

pd_all["Title+Body"] = pd_all.apply(
    lambda row: row["Title"] + ". " + row["Body"]
    if pd.notna(row["Body"])
    else row["Title"],
    axis=1,
)

# changed sentiment -> label
pd_all = pd_all.rename(
    columns={"Unnamed: 0": "id", "Labels": "label", "Title+Body": "text"}
)

# remove empty or missing labels
pd_all = pd_all[pd_all["label"].notna() & (pd_all["label"].str.strip() != "")]
pd_all = pd_all.drop_duplicates(subset=["text", "label"])
pd_all = pd_all.reset_index(drop=True)

########## 4. Configure parameters & Start training ##########

datafile = os.path.join(dataset_folder, "caffe_clean.csv")
pd_all.to_csv(datafile, index=False, columns=["id", "Number", "label", "text"])

REPEAT = 10
out_csv_name = os.path.join(script_dir, f"{project}_baseline_labels.csv")

data = pd.read_csv(datafile).fillna("")
text_col = "text"
label_col = "label"

# cleaning
data[text_col] = data[text_col].apply(remove_html)
data[text_col] = data[text_col].apply(remove_emoji)
data[text_col] = data[text_col].apply(remove_stopwords)
data[text_col] = data[text_col].apply(clean_str)

params = {"var_smoothing": np.logspace(-12, 0, 13)}
accuracies, precisions, recalls, f1_scores, auc_values = [], [], [], [], []

for repeated_time in range(REPEAT):
    indices = np.arange(data.shape[0])
    train_index, test_index = train_test_split(
        indices, test_size=0.2, random_state=repeated_time
    )

    train_text = data[text_col].iloc[train_index]
    test_text = data[text_col].iloc[test_index]
    y_train = data[label_col].iloc[train_index]
    y_test = data[label_col].iloc[test_index]

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    X_train = tfidf.fit_transform(train_text).toarray()
    X_test = tfidf.transform(test_text).toarray()

    clf = GaussianNB()
    grid = GridSearchCV(clf, params, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)

    best_clf = grid.best_estimator_
    y_pred = best_clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    try:
        fpr, tpr, _ = roc_curve(
            pd.factorize(y_test)[0], pd.factorize(y_pred)[0], pos_label=1
        )
        auc_val = auc(fpr, tpr)
    except:
        auc_val = 0.0

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)
    auc_values.append(auc_val)

final_accuracy = np.mean(accuracies)
final_precision = np.mean(precisions)
final_recall = np.mean(recalls)
final_f1 = np.mean(f1_scores)
final_auc = np.mean(auc_values)

print("=== Naive Bayes + TF-IDF (Labels) ===")
print(f"Number of repeats:     {REPEAT}")
print(f"Average Accuracy:      {final_accuracy:.4f}")
print(f"Average Precision:     {final_precision:.4f}")
print(f"Average Recall:        {final_recall:.4f}")
print(f"Average F1 score:      {final_f1:.4f}")
print(f"Average AUC:           {final_auc:.4f}")

df_log = pd.DataFrame(
    {
        "repeated_times": [REPEAT],
        "Accuracy": [final_accuracy],
        "Precision": [final_precision],
        "Recall": [final_recall],
        "F1": [final_f1],
        "AUC": [final_auc],
    }
)
df_log.to_csv(out_csv_name, mode="a", header=True, index=False)
print(f"\nResults have been saved to: {out_csv_name}")

########## 5. Custom Predictions ##########

print("\n=== Sample Predictions ===")
sample_texts = [
    "Error: Cannot compile on macOS Sonoma. The build system fails due to Apple Silicon compatibility problems.",
    "Issue: This library is not compatible with Python 3.12. The installation fails with version mismatch errors.",
    "Question: Does this framework support GPU acceleration on Windows? I only see CPU usage during training.",
]

for i, raw_text in enumerate(sample_texts):
    cleaned = clean_str(remove_stopwords(remove_emoji(remove_html(raw_text))))
    vector = tfidf.transform([cleaned]).toarray()
    prediction = best_clf.predict(vector)[0]

    print(f"\nSample #{i + 1}")
    print("Input:", raw_text.strip())
    print("Predicted Label:", prediction)