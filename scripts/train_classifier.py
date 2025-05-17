import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from models.resnet_classifier import build_resnet_classifier

def train_resnet_classifier(X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    model = build_resnet_classifier()
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint('saved_models/resnet_model.h5', save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    return model, history

if __name__ == "__main__":
    # Load preprocessed data here or from saved .npy files
    X_train = np.load('./data/processed/X_train.npy')
    y_train = np.load('./data/processed/y_train.npy')
    X_val = np.load('./data/processed/X_val.npy')
    y_val = np.load('./data/processed/y_val.npy')

    model, history = train_resnet_classifier(X_train, y_train, X_val, y_val)
