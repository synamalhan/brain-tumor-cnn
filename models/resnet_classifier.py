import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

# ResNet50 "caffe" preprocessing: RGB->BGR, subtract ImageNet channel means.
# Done with plain layers (not a Lambda) so saved weights load anywhere.
BGR_MEAN = [103.939, 116.779, 123.68]


def build_resnet_classifier(input_shape=(224, 224, 3), num_classes=1, weights='imagenet'):
    """Takes raw 0-255 RGB images. Augmentation layers are only active during training."""
    inputs = layers.Input(input_shape)
    x = layers.RandomFlip('horizontal')(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.15)(x)
    x = layers.Conv2D(3, 1, use_bias=True, trainable=False, name='rgb_to_bgr_centered',
                      kernel_initializer=tf.constant_initializer([[[[0, 0, 1], [0, 1, 0], [1, 0, 0]]]]),
                      bias_initializer=tf.constant_initializer([-m for m in BGR_MEAN]))(x)

    # Flat graph (not nested) so Grad-CAM can reach conv layers; pretrained weights copied by name
    base_model = ResNet50(weights=None, include_top=False, input_tensor=x)
    if weights:
        for ref in ResNet50(weights=weights, include_top=False, input_shape=input_shape).layers:
            if ref.weights:
                base_model.get_layer(ref.name).set_weights(ref.get_weights())
    base_model.trainable = False  # unfrozen partially by set_finetune()

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model


def grad_cam(model, x, layer='conv5_block3_out'):
    """Returns (heatmap in [0,1] at conv resolution, tumor probability) for one image batch of size 1.
    Gradients are taken w.r.t. the pre-sigmoid logit: the sigmoid saturates at confident predictions."""
    dense = model.layers[-1]
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(layer).output, model.layers[-2].output])
    with tf.GradientTape() as tape:
        conv, feats = grad_model([x], training=False)  # else Keras 3 augmentation layers stay active
        logit = (feats @ dense.kernel + dense.bias)[:, 0]
    grads = tape.gradient(logit, conv)
    cam = tf.nn.relu(tf.reduce_sum(conv * tf.reduce_mean(grads, axis=(1, 2))[:, None, None, :], -1))[0].numpy()
    return cam / (cam.max() + 1e-8), float(tf.sigmoid(logit)[0])


def set_finetune(model, prefix='conv5', lr=1e-5):
    """Unfreeze the last ResNet stage (BatchNorm stays frozen) and recompile with a small LR."""
    for layer in model.layers:
        if layer.name.startswith(prefix) and not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model
