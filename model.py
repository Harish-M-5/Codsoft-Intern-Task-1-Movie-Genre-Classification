import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------- PATH SETUP --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_FILE = os.path.join(DATASET_DIR, "train_data.txt")
TEST_FILE = os.path.join(DATASET_DIR, "test_data_solution.txt")

# -------- TEXT CLEANING --------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# -------- LOAD TRAIN DATA --------
train_texts = []
train_labels = []

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(" ::: ")
        if len(parts) == 4:
            _, title, genre, description = parts
            train_texts.append(clean_text(description))
            train_labels.append(genre)

# -------- TF-IDF --------
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X_train = tfidf.fit_transform(train_texts)

# -------- TRAIN MODEL --------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, train_labels)

# -------- LOAD TEST DATA (ACCURACY) --------
test_texts = []
test_labels = []

with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(" ::: ")
        if len(parts) == 4:
            _, title, genre, description = parts
            test_texts.append(clean_text(description))
            test_labels.append(genre)

X_test = tfidf.transform(test_texts)
predictions = model.predict(X_test)

accuracy = accuracy_score(test_labels, predictions)
print("Model Accuracy:", accuracy)

# -------- PREDICTION FUNCTION --------
def predict_genre(text):
    text = clean_text(text)
    vector = tfidf.transform([text])
    return model.predict(vector)[0]
