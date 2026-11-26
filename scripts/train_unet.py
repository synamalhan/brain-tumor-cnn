"""Train + evaluate the U-Net tumor segmenter on the Kaggle LGG MRI dataset.

    python -m scripts.train_unet [--data data/lgg/kaggle_3m]

Writes saved_models/unet_model.keras and docs/results/segmentation_*.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from models.unet_model import unet_model
from utils.preprocessing import load_segmentation_data, split_dataset

MODEL_PATH = 'saved_models/unet_model.keras'
OUT = 'docs/results'


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


def bce_dice_loss(y_true, y_pred):
    # Dice term counters the heavy background/tumor pixel imbalance
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + 1 - dice_coefficient(y_true, y_pred)


def evaluate(model, x_test, y_test, history, split_sizes):
    os.makedirs(OUT, exist_ok=True)
    pred = (model.predict(x_test, batch_size=16) > 0.5).astype(np.float32)
    t, p = y_test.reshape(len(y_test), -1), pred.reshape(len(pred), -1)
    inter, ts, ps = (t * p).sum(1), t.sum(1), p.sum(1)
    has_tumor = ts > 0
    metrics = {
        'dice_tumor_slices': float(np.mean(2 * inter[has_tumor] / (ts[has_tumor] + ps[has_tumor]))),
        'dice_global': float(2 * inter.sum() / (ts.sum() + ps.sum())),
        'iou_global': float(inter.sum() / (ts.sum() + ps.sum() - inter.sum())),
        'tumor_slice_detection_rate': float(np.mean(ps[has_tumor] > 0)),
        'empty_slice_false_positive_rate': float(np.mean(ps[~has_tumor] > 0)),
        'split_sizes': split_sizes,
    }
    json.dump(metrics, open(f'{OUT}/segmentation_metrics.json', 'w'), indent=2)
    print(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, k in zip(ax, ['loss', 'dice_coefficient']):
        a.plot(history[k], label='train')
        a.plot(history[f'val_{k}'], label='val')
        a.set_title({'loss': 'BCE + Dice loss', 'dice_coefficient': 'Dice'}[k])
        a.set_xlabel('epoch')
        a.legend()
    plt.tight_layout()
    plt.savefig(f'{OUT}/segmentation_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Examples: the 5 test slices with the largest tumors (stable, easy to read)
    idx = np.argsort(-ts)[:5]
    fig, ax = plt.subplots(3, 5, figsize=(16, 10))
    for j, i in enumerate(idx):
        d = 2 * inter[i] / (ts[i] + ps[i])
        for r, (m, title) in enumerate([(None, 'MRI (T1 / FLAIR / T1-Gd)'), (y_test[i], 'Ground truth'), (pred[i], f'U-Net (Dice {d:.2f})')]):
            ax[r, j].imshow(x_test[i])
            if m is not None:
                ax[r, j].imshow(np.ma.masked_where(m[..., 0] == 0, m[..., 0]), cmap='autumn', alpha=0.55)
            ax[r, j].set_title(title)
            ax[r, j].axis('off')
    plt.suptitle('U-Net on held-out test patients')
    plt.tight_layout()
    plt.savefig(f'{OUT}/segmentation_examples.png', dpi=100, bbox_inches='tight')
    plt.close()
    return metrics


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/lgg/kaggle_3m')
    ap.add_argument('--base-filters', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=40)
    args = ap.parse_args()

    print('Loading data...')
    images, masks, patients = load_segmentation_data(args.data)
    images = images.astype(np.float32) / 255.0
    masks = (masks[..., None] > 127).astype(np.float32)

    # Split by patient so no patient's slices appear in both train and test
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_dataset(
        images, masks, patients, test_size=0.1, val_size=0.1)
    sizes = {'train': len(x_train), 'val': len(x_val), 'test': len(x_test),
             'patients': int(len(set(patients)))}
    print(sizes)

    model = unet_model(input_size=(256, 256, 3), base_filters=args.base_filters)
    model.compile(optimizer=Adam(1e-3), loss=bce_dice_loss, metrics=[dice_coefficient, iou])

    os.makedirs('saved_models', exist_ok=True)
    h = model.fit(
        x_train, y_train, validation_data=(x_val, y_val), epochs=args.epochs, batch_size=16,
        callbacks=[ModelCheckpoint(MODEL_PATH, monitor='val_dice_coefficient', mode='max', save_best_only=True),
                   ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3),
                   EarlyStopping(monitor='val_dice_coefficient', mode='max', patience=8, restore_best_weights=True)])
    evaluate(model, x_test, y_test, h.history, sizes)
