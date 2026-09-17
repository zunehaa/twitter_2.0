import pandas as pd
import numpy as np
import re
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import os

print("Loading data...")
df = pd.read_csv("data/twitter_training.csv", header=None, names=["id", "entity", "sentiment", "text"])
df = df.dropna(subset=["text"]).reset_index(drop=True)
df = df.drop_duplicates(subset=["text", "sentiment"]).reset_index(drop=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Cleaning text...")
df["clean_text"] = df["text"].apply(clean_text)
df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

le = LabelEncoder()
df["label"] = le.fit_transform(df["sentiment"])
num_classes = len(le.classes_)

X = df["clean_text"].values
y = to_categorical(df["label"].values, num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df["label"].values)

VOCAB_SIZE = 20000
MAX_LEN = 40

print("Tokenizing...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")

EMBED_DIM = 128

model = Sequential([
    Input(shape=(MAX_LEN,)),
    Embedding(VOCAB_SIZE, EMBED_DIM),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(128, dropout=0.2, return_sequences=True)),
    LSTM(64, dropout=0.2),
    Dense(64, activation="relu"),
    Dense(num_classes, activation="softmax"),
])
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print("Training model... this might take a few minutes...")
early_stop = EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)

model.fit(
    X_train_pad, y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

os.makedirs("model", exist_ok=True)
model.save("model/lstm_model.h5")
with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Saved all model files successfully.")
