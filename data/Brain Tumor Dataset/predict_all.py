
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

H = 256
W = 256

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


""" Create a directory. """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path, split=0.1):
    x = sorted(glob(f"{path}/image/*"))
    y = sorted(glob(f"{path}/mask/*"))

    split_size = int(len(x) * split)

    train_x, valid_x = train_test_split(x, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(y, test_size=split_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    train_x, train_y = shuffle(train_x, train_y, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def process_mask(x):
    x = np.squeeze(x, axis=-1)
    x = x > 0.5
    x = x.astype(np.float32)
    x = x * 255.0
    x = np.expand_dims(x, axis=-1)
    x = np.concatenate([x, x, x], axis=-1)
    return x

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results")

    """ Dataset path """
    dataset_path = "/media/nikhil/ML/ml_dataset/brain tumor dataset/data"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)
    print(f"Test Images: {len(test_x)} - Test Masks: {len(test_y)}")

    """ Loading the model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        unet_model = tf.keras.models.load_model("UNET/files/model.h5")
        resunet_model = tf.keras.models.load_model("RESUNET/files/model.h5")
        deeplabv3plus_model = tf.keras.models.load_model("DEEPLABV3PLUS/files/model.h5")

    """ Loop over the test dataset """
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extracing the image name. """
        image_name = x.split("/")[-1]

        """ Reading image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (H, W))
        x = image/255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        """ Reading Mask """
        mask = cv2.imread(y, cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, (H, W))

        """ Prediction """
        unet_mask = process_mask(unet_model.predict(x)[0])
        resunet_mask = process_mask(resunet_model.predict(x)[0])
        deeplabv3plus_mask = process_mask(deeplabv3plus_model.predict(x)[0])

        """ Save the image, gt, predicted masks """
        line = np.ones((H, 10, 3)) * 255.0
        cat_image = np.concatenate([
            image, line,
            mask, line,
            unet_mask, line,
            resunet_mask, line,
            deeplabv3plus_mask
        ], axis=1)

        cv2.imwrite(f"results/{image_name}", cat_image)
