"""Train a binary chest X-ray pneumonia classifier.

Expected dataset layout:

chest_xray/
    train/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/

Example:
    python chest_xray_train.py --data-dir chest_xray --epochs 15
"""

from __future__ import annotations

import argparse
import json
import importlib
from pathlib import Path

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42


def get_tensorflow():
    try:
        return importlib.import_module("tensorflow")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "TensorFlow is not installed. Install it with: pip install tensorflow"
        ) from error


def build_datasets(tf, data_dir: Path):
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    class_names = sorted([item.name for item in train_dir.iterdir() if item.is_dir()])

    def count_images(class_name: str) -> int:
        return sum(1 for item in (train_dir / class_name).iterdir() if item.is_file())

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = tf.cast(image, tf.float32)
        return image, tf.cast(label, tf.float32)

    def make_class_dataset(class_name: str, label: float, target_count: int):
        file_paths = sorted([item for item in (train_dir / class_name).iterdir() if item.is_file()])
        path_ds = tf.data.Dataset.from_tensor_slices([str(path) for path in file_paths])
        path_ds = path_ds.shuffle(len(file_paths), seed=SEED, reshuffle_each_iteration=True).repeat()
        label_ds = tf.data.Dataset.from_tensors(tf.cast(label, tf.float32)).repeat()
        ds = tf.data.Dataset.zip((path_ds, label_ds))
        ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.take(target_count)

    class_counts = {class_name: count_images(class_name) for class_name in class_names}
    target_count = max(class_counts.values())

    balanced_train_ds = None
    for index, class_name in enumerate(class_names):
        class_ds = make_class_dataset(class_name, float(index), target_count)
        balanced_train_ds = class_ds if balanced_train_ds is None else balanced_train_ds.concatenate(class_ds)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="binary",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = balanced_train_ds.shuffle(target_count * len(class_names), seed=SEED, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    test_ds = test_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, test_ds, class_names, class_counts


def build_model(tf):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            data_augmentation,
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a chest X-ray pneumonia classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("chest_xray"), help="Path to the chest_xray dataset folder")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory to save models and metrics")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    tf = get_tensorflow()

    train_ds, val_ds, test_ds, class_names, class_counts = build_datasets(tf, data_dir)

    model = build_model(tf)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    test_results = model.evaluate(test_ds, verbose=1)
    test_metrics = dict(zip(model.metrics_names, [float(value) for value in test_results]))

    model.save(output_dir / "final_model.keras")

    with open(output_dir / "class_names.json", "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    with open(output_dir / "history.json", "w", encoding="utf-8") as file:
        json.dump(history.history, file, indent=2)

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(test_metrics, file, indent=2)

    print("Training complete.")
    print(f"Classes: {class_names}")
    print(f"Class counts: {class_counts}")
    print(f"Test metrics: {test_metrics}")
    print(f"Saved artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
