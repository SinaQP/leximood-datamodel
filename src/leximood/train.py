"""Training pipeline for the LexiMood emotion classification model.

This module mirrors the steps used in ``LexiMood.ipynb`` and packages them into
reusable functions so the workflow is easier to maintain and run from a CLI.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for tokenization and model training."""

    vocab_size: int = 15_000
    max_len: int = 100
    embedding_dim: int = 128
    lstm_units: int = 128
    epochs: int = 15
    batch_size: int = 64
    test_size: float = 0.2
    random_state: int = 42
    oov_token: str = "<UNK>"


def load_dataset(csv_path: Path):
    """Load and validate the input dataset.

    The CSV is expected to contain ``text`` and ``emotion`` columns.
    """

    import pandas as pd

    df = pd.read_csv(csv_path)
    required_columns = {"text", "emotion"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df


def prepare_data(df, config: TrainConfig):
    """Encode labels and build train/test splits with padded token sequences."""

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import to_categorical

    encoder = LabelEncoder()
    integer_labels = encoder.fit_transform(df["emotion"])
    emotion_labels = encoder.classes_
    num_classes = len(emotion_labels)
    one_hot_labels = to_categorical(integer_labels, num_classes=num_classes)

    texts = df["text"].values
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        one_hot_labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=integer_labels,
    )

    tokenizer = Tokenizer(num_words=config.vocab_size, oov_token=config.oov_token)
    tokenizer.fit_on_texts(x_train)

    train_sequences = tokenizer.texts_to_sequences(x_train)
    test_sequences = tokenizer.texts_to_sequences(x_test)

    train_padded = pad_sequences(
        train_sequences,
        maxlen=config.max_len,
        padding="post",
        truncating="post",
    )
    test_padded = pad_sequences(
        test_sequences,
        maxlen=config.max_len,
        padding="post",
        truncating="post",
    )

    return train_padded, test_padded, y_train, y_test, tokenizer, emotion_labels


def build_model(config: TrainConfig, num_classes: int):
    """Build the BiLSTM architecture from the original notebook."""

    import tensorflow as tf

    inputs = tf.keras.Input(shape=(config.max_len,))
    x = tf.keras.layers.Embedding(config.vocab_size, config.embedding_dim)(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(config.lstm_units, return_sequences=True)
    )(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(csv_path: Path, config: TrainConfig):
    """Run end-to-end training and return the trained model/history tuple."""

    df = load_dataset(csv_path)
    train_padded, test_padded, y_train, y_test, _, emotion_labels = prepare_data(df, config)

    model = build_model(config, num_classes=len(emotion_labels))
    history = model.fit(
        train_padded,
        y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=(test_padded, y_test),
        verbose=1,
    )

    return model, history, emotion_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LexiMood emotion model.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("emotion_dataset.csv"),
        help="Path to emotion dataset CSV containing text and emotion columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig()
    train(args.dataset, config)


if __name__ == "__main__":
    main()
