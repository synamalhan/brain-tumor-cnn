import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_data(csv_path, images_dir):
    """
    Load image paths and labels from CSV and image folder.
    """
    df = pd.read_csv(csv_path)
    df['path'] = df['image'].apply(lambda x: os.path.join(images_dir, x))
    df['label'] = df['class'].apply(lambda x: 1 if x.lower() == 'tumor' else 0)
    return df[['path', 'label']] 

def preprocess_image(img_path, target_size=(256, 256)):
    """
    Open and preprocess a single image.
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return img_array.astype(np.float32)

def create_dataset(df, target_size=(256, 256)):
    """
    Preprocess all images and return arrays.
    """
    X = np.array([preprocess_image(p, target_size) for p in df['path']])
    y = df['label'].values
    return X, y

def split_dataset(X, y, test_size=0.3, val_split=0.5, seed=42):
    """
    Split into train, val, and test sets.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_split, random_state=seed, stratify=y_temp)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
