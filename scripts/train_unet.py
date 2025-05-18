import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

from models.unet_model import unet_model
from utils.preprocessing import load_segmentation_data  # You must implement this

# Optional: Dice coefficient and IoU
def dice_coefficient(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-7) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1e-7)

def iou(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + 1e-7) / (union + 1e-7)


def train_unet():
    print("Loading data...")
    images, masks = load_segmentation_data(
        image_dir='data/segmentation/images',
        mask_dir='data/segmentation/masks',
        target_size=(256, 256)
    )

    print("Normalizing and binarizing...")
    images = images / 255.0
    masks = masks / 255.0
    masks = (masks > 0.5).astype(np.float32)

    print("Splitting train/val data...")
    x_train, x_val, y_train, y_val = train_test_split(
        images, masks, test_size=0.2, random_state=42, shuffle=True
    )

    print(f"Training samples: {x_train.shape[0]}, Validation samples: {x_val.shape[0]}")

    print("Building model...")
    model = unet_model(input_size=(256, 256, 3))
    model.compile(optimizer=Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy', dice_coefficient, iou])

    model.summary()

    os.makedirs("saved_models", exist_ok=True)

    checkpoint_cb = ModelCheckpoint("saved_models/unet_model.h5",
                                    save_best_only=True,
                                    verbose=1)
    early_stop_cb = EarlyStopping(patience=10, restore_best_weights=True, verbose=1)
    csv_logger = CSVLogger("saved_models/training_log.csv", append=True)

    print("Starting training...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=50,
        batch_size=16,
        callbacks=[checkpoint_cb, early_stop_cb, csv_logger]
    )

    # Optional: plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coefficient'], label='Train Dice')
    plt.plot(history.history['val_dice_coefficient'], label='Val Dice')
    plt.title("Dice Coefficient")
    plt.legend()

    plt.tight_layout()
    plt.savefig("saved_models/training_curves.png")
    plt.show()

if __name__ == "__main__":
    train_unet()
