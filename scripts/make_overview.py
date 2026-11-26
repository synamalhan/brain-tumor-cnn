"""Build docs/overview.png: one image with headline results, plots and app screenshots.

    python scripts/make_overview.py   (after training + taking UI screenshots into docs/ui/)
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

R, UI = 'docs/results', 'docs/ui'
cls = json.load(open(f'{R}/metrics.json'))
seg = json.load(open(f'{R}/segmentation_metrics.json'))
PURPLE, PINK, INK, MUTED = '#6d28d9', '#db2777', '#1f2937', '#6b7280'


def img(ax, path, title):
    ax.imshow(plt.imread(path))
    ax.set_title(title, fontsize=16, color=INK, pad=8, fontweight='bold')
    ax.axis('off')


fig = plt.figure(figsize=(22, 31), facecolor='white')
gs = fig.add_gridspec(6, 6, height_ratios=[1.1, 1.2, 6.1, 3.9, 8.3, 9.2], hspace=0.16, wspace=0.08,
                      left=0.03, right=0.97, top=0.985, bottom=0.01)

# header band with gradient
ax = fig.add_subplot(gs[0, :])
grad = np.linspace(0, 1, 512)[None, :]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('brand', ['#312e81', PURPLE, PINK])
ax.imshow(grad, aspect='auto', cmap=cmap, extent=(0, 1, 0, 1))
ax.text(0.02, 0.62, 'NeuroScan · Brain Tumor Detection & Segmentation', color='white',
        fontsize=30, fontweight='bold', va='center', transform=ax.transAxes)
ax.text(0.02, 0.25, 'ResNet50 classifier with Grad-CAM explanations · U-Net tumor segmentation · Streamlit app',
        color='white', alpha=.9, fontsize=17, va='center', transform=ax.transAxes)
ax.axis('off')

# metric cards
ax = fig.add_subplot(gs[1, :])
ax.set_xlim(0, 8); ax.set_ylim(0, 1); ax.axis('off')
cards = [('Accuracy', f"{cls['accuracy']:.1%}", PURPLE), ('ROC AUC', f"{cls['roc_auc']:.3f}", PURPLE),
         ('Sensitivity', f"{cls['recall_tumor']:.1%}", PURPLE), ('Specificity', f"{cls['specificity']:.1%}", PURPLE),
         ('F1 (tumor)', f"{cls['f1_tumor']:.3f}", PURPLE),
         ('Dice', f"{seg['dice_global']:.3f}", PINK), ('IoU', f"{seg['iou_global']:.3f}", PINK),
         ('Found tumor slices', f"{seg['tumor_slice_detection_rate']:.0%}", PINK)]
for i, (label, value, color) in enumerate(cards):
    ax.add_patch(FancyBboxPatch((i + .06, .08), .88, .84, boxstyle='round,pad=0,rounding_size=.08',
                                fc=matplotlib.colors.to_rgba(color, .07), ec=matplotlib.colors.to_rgba(color, .45), lw=1.5))
    ax.text(i + .5, .66, label, ha='center', fontsize=14, color=MUTED)
    ax.text(i + .5, .3, value, ha='center', fontsize=24, color=color, fontweight='bold')
ax.text(0.06, -0.12, f"Classifier: {cls['split_sizes']['test']} held-out images, duplicate-aware split",
        fontsize=12, color=PURPLE)
ax.text(5.06, -0.12, f"Segmenter: {seg['split_sizes']['test']} slices from held-out patients",
        fontsize=12, color=PINK)

# app screenshots
img(fig.add_subplot(gs[2, :3]), f'{UI}/detection_tumor.png', 'App · Tumor detection with Grad-CAM')
img(fig.add_subplot(gs[2, 3:]), f'{UI}/segmentation.png', 'App · Segmentation vs. expert mask')

# classifier plots
img(fig.add_subplot(gs[3, 0]), f'{R}/confusion_matrix.png', 'Confusion matrix')
img(fig.add_subplot(gs[3, 1]), f'{R}/roc_curve.png', 'ROC curve')
img(fig.add_subplot(gs[3, 2:4]), f'{R}/training_curves.png', 'Classifier training')
img(fig.add_subplot(gs[3, 4:]), f'{R}/segmentation_curves.png', 'U-Net training')

# qualitative results
img(fig.add_subplot(gs[4, :]), f'{R}/gradcam_examples.png', 'Grad-CAM on held-out scans: the model attends to the lesion')
img(fig.add_subplot(gs[5, :]), f'{R}/segmentation_examples.png', 'U-Net on held-out patients: ground truth vs. prediction')

fig.savefig('docs/overview.png', dpi=80, facecolor='white')
print('wrote docs/overview.png')
