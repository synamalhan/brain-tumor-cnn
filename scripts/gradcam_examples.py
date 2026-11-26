"""Regenerate docs/results/gradcam_examples.png:  python scripts/gradcam_examples.py"""
import sys; sys.path.insert(0, '.')
import numpy as np, tensorflow as tf, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from models.resnet_classifier import build_resnet_classifier, grad_cam
from utils.preprocessing import load_classification_data, split_dataset
X, y, g = load_classification_data('data/download/Brain Tumor Data Set/Brain Tumor Data Set')
_, _, (Xt, yt) = split_dataset(X, y, g)
m = build_resnet_classifier(weights=None); m.load_weights('saved_models/resnet_classifier.weights.h5')
rng = np.random.default_rng(0)
idx = list(rng.choice(np.where(yt == 1)[0], 4, replace=False)) + list(rng.choice(np.where(yt == 0)[0], 2, replace=False))
fig, ax = plt.subplots(2, 6, figsize=(18, 7))
for j, i in enumerate(idx):
    x = Xt[i:i+1].astype('float32')
    cam, prob = grad_cam(m, x)
    cam = tf.image.resize(cam[..., None] / (cam.max() + 1e-8), (224, 224))[..., 0].numpy()
    ax[0, j].imshow(Xt[i]); ax[0, j].set_title(f"True: {'Tumor' if yt[i] else 'Healthy'}")
    ax[1, j].imshow(Xt[i]); ax[1, j].imshow(cam, cmap='inferno', alpha=0.45)
    ax[1, j].set_title(f"P(tumor) = {prob:.2f}")
for a in ax.ravel(): a.axis('off')
plt.suptitle('Grad-CAM on held-out test scans (top: input, bottom: evidence for "tumor")')
plt.tight_layout(h_pad=2.5)
assert np.allclose(grad_cam(m, Xt[idx[:1]].astype("float32"))[0], grad_cam(m, Xt[idx[:1]].astype("float32"))[0])  # Grad-CAM must be deterministic
plt.savefig('docs/results/gradcam_examples.png', dpi=110, bbox_inches='tight')
