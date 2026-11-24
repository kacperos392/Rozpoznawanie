from tensorflow import keras
from tensorflow.keras import layers

from src.config import INPUT_SHAPE, NUM_CLASSES


def conv_block(x, filters):
    """Blok: Conv2D -> BatchNorm -> ReLU -> MaxPooling2D."""
    x = layers.Conv2D(filters, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    return x


def build_model(dropout_rate=0.4):
    inputs = keras.Input(shape=INPUT_SHAPE)

    x = conv_block(inputs, 16)
    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="music_genre_cnn")
    return model


def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model