"""Train + evaluate the ResNet50 tumor classifier.

    python -m scripts.train_classifier [--data "data/download/Brain Tumor Data Set/Brain Tumor Data Set"]

Writes saved_models/resnet_classifier.weights.h5 and docs/results/*.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve,
                             ConfusionMatrixDisplay)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from models.resnet_classifier import build_resnet_classifier, set_finetune
from utils.preprocessing import load_classification_data, split_dataset

WEIGHTS = 'saved_models/resnet_classifier.weights.h5'
OUT = 'docs/results'


def train_resnet_classifier(X_train, y_train, X_val, y_val, head_epochs=8, ft_epochs=15, batch_size=32):
    model = build_resnet_classifier()
    ckpt = ModelCheckpoint(WEIGHTS, monitor='val_auc', mode='max', save_best_only=True, save_weights_only=True)

    # Phase 1: train only the new head on frozen ImageNet features
    h1 = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=head_epochs,
                   batch_size=batch_size, callbacks=[ckpt])
    # Phase 2: fine-tune the last ResNet stage
    set_finetune(model)
    h2 = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=ft_epochs, batch_size=batch_size,
                   callbacks=[ckpt, ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2),
                              EarlyStopping(monitor='val_auc', mode='max', patience=5)])
    model.load_weights(WEIGHTS)
    history = {k: h1.history[k] + h2.history[k] for k in h1.history if k in h2.history}
    return model, history, head_epochs


def evaluate(model, X_test, y_test, history, head_epochs, split_sizes):
    os.makedirs(OUT, exist_ok=True)
    probs = model.predict(X_test, batch_size=32).ravel()
    pred = (probs > 0.5).astype(int)
    rep = classification_report(y_test, pred, target_names=['healthy', 'tumor'], output_dict=True)
    metrics = {
        'accuracy': rep['accuracy'],
        'precision_tumor': rep['tumor']['precision'],
        'recall_tumor': rep['tumor']['recall'],
        'f1_tumor': rep['tumor']['f1-score'],
        'specificity': rep['healthy']['recall'],
        'roc_auc': roc_auc_score(y_test, probs),
        'confusion_matrix': confusion_matrix(y_test, pred).tolist(),
        'split_sizes': split_sizes,
    }
    json.dump(metrics, open(f'{OUT}/metrics.json', 'w'), indent=2)
    print(classification_report(y_test, pred, target_names=['healthy', 'tumor'], digits=4))
    print(json.dumps(metrics, indent=2))

    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=['Healthy', 'Tumor'], cmap='Blues')
    plt.title('Test confusion matrix')
    plt.savefig(f'{OUT}/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, lw=2, label=f'ResNet50 (AUC = {metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], '--', color='grey')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Test ROC curve')
    plt.legend(loc='lower right')
    plt.savefig(f'{OUT}/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, k in zip(ax, ['loss', 'accuracy']):
        a.plot(history[k], label='train')
        a.plot(history[f'val_{k}'], label='val')
        a.axvline(head_epochs - 0.5, ls=':', color='grey', label='start fine-tuning')
        a.set_title(k.capitalize())
        a.set_xlabel('epoch')
        a.legend()
    plt.tight_layout()
    plt.savefig(f'{OUT}/training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    return metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/download/Brain Tumor Data Set/Brain Tumor Data Set')
    args = p.parse_args()

    os.makedirs('saved_models', exist_ok=True)
    X, y, groups = load_classification_data(args.data)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(X, y, groups)
    sizes = {'train': len(y_train), 'val': len(y_val), 'test': len(y_test),
             'total_images': len(y), 'unique_scans': len(set(groups))}
    print(sizes)

    model, history, head_epochs = train_resnet_classifier(X_train, y_train, X_val, y_val)
    evaluate(model, X_test, y_test, history, head_epochs, sizes)
