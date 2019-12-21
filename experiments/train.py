import os
from typing import List
from typing import Tuple

import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

ROWS = 300000  # Set to None to load all rows.
EPOCHS = 5
VALIDATION_SPLIT = 0.05
BATCH_SIZE = 64
EMBEDDING_DIM = 32
MAX_NUM_WORDS = 30000
MAX_SEQUENCE_LENGTH = 384


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

TRAIN_DATA_PATH = os.path.join(root_dir, "data", "processed", "train.csv")
TEST_DATA_PATH = os.path.join(root_dir, "data", "processed", "test.csv")


def load_data(path: str, nrows: int) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(path, nrows=nrows)
    df.fillna("", inplace=True)
    samples = df.title + " " + df.review
    labels = df.label
    return samples.values, labels.values


def compile_model() -> keras.Sequential:
    model = keras.models.Sequential(
        [
            keras.layers.Embedding(
                input_dim=MAX_NUM_WORDS,
                input_length=MAX_SEQUENCE_LENGTH,
                output_dim=EMBEDDING_DIM,
            ),
            keras.layers.Conv1D(filters=64, kernel_size=5, activation="relu"),
            keras.layers.MaxPooling1D(4),
            keras.layers.LSTM(units=128),
            keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(units=256, activation="relu"),
            keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(units=1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


def train() -> None:
    train_samples, train_labels = load_data(path=TRAIN_DATA_PATH, nrows=ROWS)

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts=train_samples)

    train_sequences = tokenizer.texts_to_sequences(train_samples)
    padded_train_sequences = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    model = compile_model()
    model.fit(
        padded_train_sequences,
        train_labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=2),
            keras.callbacks.TensorBoard(update_freq="epoch", embeddings_freq=1),
            keras.callbacks.ModelCheckpoint(
                filepath="models/model{epoch:02d}.hdf5", save_best_only=True
            ),
        ],
    )

    test_samples, test_labels = load_data(path=TEST_DATA_PATH, nrows=50000)
    test_sequences = tokenizer.texts_to_sequences(test_samples)
    padded_test_sequences = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    test_metrics = model.evaluate(padded_test_sequences, test_labels, verbose=0)
    print(f"\nTest metrics: {test_metrics}")


if __name__ == "__main__":
    train()
