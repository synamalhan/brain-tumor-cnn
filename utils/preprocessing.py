import os
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

IMG_SIZE = (224, 224)
CLASS_DIRS = {'Healthy': 0, 'Brain Tumor': 1}


def image_hash(img):
    """16x16 average hash: identical for re-encoded / resized copies of the same scan."""
    p = np.asarray(img.convert('L').resize((16, 16)), dtype=np.float32)
    return (p > p.mean()).tobytes()


def load_classification_data(root, target_size=IMG_SIZE):
    """Load the Kaggle 'Brain Tumor Data Set' folder layout.
    Returns uint8 images (0-255), labels and duplicate-group ids."""
    images, labels, groups, seen = [], [], [], {}
    for cls, label in CLASS_DIRS.items():
        d = os.path.join(root, cls)
        for f in sorted(os.listdir(d)):
            try:
                img = Image.open(os.path.join(d, f)).convert('RGB')
            except OSError:
                continue  # skip non-image / corrupt files
            groups.append(seen.setdefault(image_hash(img), len(seen)))
            images.append(np.asarray(img.resize(target_size), dtype=np.uint8))
            labels.append(label)
    return np.stack(images), np.array(labels), np.array(groups)


def split_dataset(X, y, groups, test_size=0.15, val_size=0.15, seed=42):
    """70/15/15 split where near-duplicate images never cross splits (prevents leakage)."""
    trainval, test = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(X, y, groups))
    rel_val = val_size / (1 - test_size)
    tr, va = next(GroupShuffleSplit(1, test_size=rel_val, random_state=seed).split(trainval, groups=groups[trainval]))
    train, val = trainval[tr], trainval[va]
    assert not set(groups[train]) & set(groups[test]) and not set(groups[val]) & set(groups[test])
    return (X[train], y[train]), (X[val], y[val]), (X[test], y[test])


def load_segmentation_data(root, target_size=(256, 256)):
    """Load the Kaggle LGG layout: root/<patient>/<slice>.tif + <slice>_mask.tif.
    Returns uint8 images, uint8 masks (0/255) and patient ids for group splitting."""
    images, masks, groups = [], [], []
    for pid, patient in enumerate(sorted(p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p)))):
        d = os.path.join(root, patient)
        for f in sorted(os.listdir(d)):
            if not f.endswith('.tif') or f.endswith('_mask.tif'):
                continue
            images.append(np.asarray(Image.open(os.path.join(d, f)).convert('RGB').resize(target_size)))
            mask_file = os.path.join(d, f[:-4] + '_mask.tif')
            masks.append(np.asarray(Image.open(mask_file).convert('L').resize(target_size, Image.NEAREST)))
            groups.append(pid)
    return np.stack(images), np.stack(masks), np.array(groups)
