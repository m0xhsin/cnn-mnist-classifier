#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from keras import layers, models


def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model


def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("python3 cnn.py <data.npz> train|test")
        sys.exit(1)

    npz_path = sys.argv[1]
    mode = sys.argv[2]

    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]

    num_classes = len(np.unique(y))
    input_shape = X.shape[1:]

    print("Loaded data:")
    print("X:", X.shape)
    print("y:", y.shape)
    print("Classes:", num_classes)

    model = build_cnn(input_shape, num_classes)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    if mode == "train":
        # 80/20 split ONLY for training
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model.summary()

        model.fit(
            X_train, y_train,
            epochs=15,
            batch_size=32,
            validation_data=(X_val, y_val)
        )

        model.save_weights("model.weights.h5")
        print("Model weights saved to model.weights.h5")

    elif mode == "test":
        model.load_weights("model.weights.h5")
        print("Loaded model weights")

        test_loss, test_acc = model.evaluate(X, y, verbose=0)
        print(f"Test accuracy: {test_acc:.4f}")

        y_pred = model.predict(X)
        y_pred = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.show()

    else:
        print("Second argument must be 'train' or 'test'")
        sys.exit(1)


if __name__ == "__main__":
    main()
