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
PURPLE, PINK, INK, MUTED, RULE = '#0e7490', '#be123c', '#0f172a', '#64748b', '#e2e8f0'  # teal (classifier), rose (segmenter)
MONO = 'DejaVu Sans Mono'


def img(ax, path, title):
    ax.imshow(plt.imread(path))
    ax.set_title(title, fontsize=15, color=INK, pad=8, family=MONO, loc='left')
    ax.axis('off')


fig = plt.figure(figsize=(22, 31), facecolor='white')
gs = fig.add_gridspec(6, 6, height_ratios=[1.1, 1.2, 6.1, 3.9, 8.3, 9.2], hspace=0.16, wspace=0.08,
                      left=0.03, right=0.97, top=0.985, bottom=0.01)

# plain header: repo name, title, rule
ax = fig.add_subplot(gs[0, :])
ax.text(0, 0.86, 'brain-tumor-cnn / NeuroScan', family=MONO, fontsize=14, color=MUTED, va='center', transform=ax.transAxes)
ax.text(1, 0.86, 'TensorFlow · ResNet50 · U-Net · Grad-CAM · Streamlit', family=MONO, fontsize=14, color=MUTED,
        ha='right', va='center', transform=ax.transAxes)
ax.axhline(0.7, color=RULE, lw=1.2)
ax.text(0, 0.38, 'Brain tumor detection & segmentation from MRI', color=INK, fontsize=30, fontweight='bold',
        va='center', transform=ax.transAxes)
ax.text(0, 0.02, 'all metrics on held-out data · duplicate-aware split (classifier) · patient-level split (segmenter)',
        family=MONO, color=MUTED, fontsize=14, va='center', transform=ax.transAxes)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

# metric cards
ax = fig.add_subplot(gs[1, :])
ax.set_xlim(0, 8); ax.set_ylim(0, 1); ax.axis('off')
cards = [('accuracy', f"{cls['accuracy']:.3f}", PURPLE), ('roc auc', f"{cls['roc_auc']:.3f}", PURPLE),
         ('sensitivity', f"{cls['recall_tumor']:.3f}", PURPLE), ('specificity', f"{cls['specificity']:.3f}", PURPLE),
         ('f1 (tumor)', f"{cls['f1_tumor']:.3f}", PURPLE),
         ('dice', f"{seg['dice_global']:.3f}", PINK), ('iou', f"{seg['iou_global']:.3f}", PINK),
         ('tumor slices found', f"{seg['tumor_slice_detection_rate']:.3f}", PINK)]
for i, (label, value, color) in enumerate(cards):
    ax.add_patch(FancyBboxPatch((i + .06, .08), .88, .84, boxstyle='round,pad=0,rounding_size=.03',
                                fc='#fafafa', ec=RULE, lw=1.2))
    ax.plot([i + .06, i + .06], [.08, .92], color=color, lw=3, solid_capstyle='butt')  # accent bar
    ax.text(i + .14, .64, label, family=MONO, fontsize=13, color=MUTED)
    ax.text(i + .14, .26, value, family=MONO, fontsize=23, color=INK)
ax.text(0.06, -0.12, f"classifier · n={cls['split_sizes']['test']} test images", family=MONO, fontsize=12, color=PURPLE)
ax.text(5.06, -0.12, f"segmenter · {seg['split_sizes']['test']} slices, held-out patients", family=MONO, fontsize=12, color=PINK)

# app screenshots
img(fig.add_subplot(gs[2, :3]), f'{UI}/detection_tumor.png', '(a) app: detection with Grad-CAM')
img(fig.add_subplot(gs[2, 3:]), f'{UI}/segmentation.png', '(b) app: segmentation vs. expert mask')

# classifier plots
img(fig.add_subplot(gs[3, 0]), f'{R}/confusion_matrix.png', '(c) confusion matrix')
img(fig.add_subplot(gs[3, 1]), f'{R}/roc_curve.png', '(d) ROC curve')
img(fig.add_subplot(gs[3, 2:4]), f'{R}/training_curves.png', '(e) classifier training')
img(fig.add_subplot(gs[3, 4:]), f'{R}/segmentation_curves.png', '(f) U-Net training')

# qualitative results
img(fig.add_subplot(gs[4, :]), f'{R}/gradcam_examples.png', '(g) Grad-CAM on held-out scans')
img(fig.add_subplot(gs[5, :]), f'{R}/segmentation_examples.png', '(h) U-Net on held-out patients: ground truth vs. prediction')

fig.savefig('docs/overview.png', dpi=80, facecolor='white')
print('wrote docs/overview.png')
