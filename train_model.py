import os
import numpy as np
import tensorflow as tf

from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

print("========================================")
print(" FINGERPRINT-X V2 TRAINING")
print("========================================")

# -----------------------------
# SETTINGS
# -----------------------------

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/validation"

MODEL_PATH = "fingerprint_gender_model_v2.keras"

# -----------------------------
# LABEL FUNCTION
# -----------------------------

def get_label(filename):

    parts = filename.split("__")[1]
    gender = parts.split("_")[0]

    if gender == "M":
        return 1

    if gender == "F":
        return 0

    return None


# -----------------------------
# LOAD FILES
# -----------------------------

train_files = [
    f for f in os.listdir(TRAIN_DIR)
    if f.lower().endswith(".bmp")
]

val_files = [
    f for f in os.listdir(VAL_DIR)
    if f.lower().endswith(".bmp")
]

print("Training images:", len(train_files))
print("Validation images:", len(val_files))


# -----------------------------
# LOAD DATA
# -----------------------------

def load_dataset(directory, files):

    images = []
    labels = []

    for filename in files:

        label = get_label(filename)

        if label is None:
            continue

        path = os.path.join(directory, filename)

        image = tf.keras.utils.load_img(
            path,
            color_mode="grayscale",
            target_size=IMG_SIZE
        )

        image = tf.keras.utils.img_to_array(image)

        image = image / 255.0

        images.append(image)
        labels.append(label)

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.int32)
    )


print("Loading training data...")

X_train, y_train = load_dataset(
    TRAIN_DIR,
    train_files
)

print("Loading validation data...")

X_val, y_val = load_dataset(
    VAL_DIR,
    val_files
)

print("========================================")
print("DATA READY")
print("========================================")

print("Train:", X_train.shape)
print("Validation:", X_val.shape)

print("Train labels:", Counter(y_train))
print("Validation labels:", Counter(y_val))


# -----------------------------
# CLASS WEIGHTS
# -----------------------------

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(c): float(w)
    for c, w in zip(classes, weights)
}

print("Class weights:", class_weights)


# -----------------------------
# DATA AUGMENTATION
# -----------------------------

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    )
])


# -----------------------------
# IMPROVED CNN
# -----------------------------

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(128, 128, 1)
    ),

    data_augmentation,

    # Block 1
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(),

    # Block 2
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(),

    # Block 3
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(),

    # Reduce parameters
    tf.keras.layers.GlobalAveragePooling2D(),

    # Classifier
    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# -----------------------------
# COMPILE
# -----------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(
            name="auc"
        )
    ]
)


print("========================================")
print("MODEL SUMMARY")
print("========================================")

model.summary()


# -----------------------------
# CALLBACKS
# -----------------------------

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        patience=3,
        mode="max",
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001
    ),

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_auc",
        mode="max",
        save_best_only=True
    )
]


# -----------------------------
# TRAIN
# -----------------------------

print("========================================")
print("TRAINING STARTED")
print("========================================")

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1
)


# -----------------------------
# FINAL INFO
# -----------------------------

print("========================================")
print("TRAINING COMPLETE")
print("========================================")

print(
    "Best validation AUC:",
    max(history.history["val_auc"])
)

print(
    "Best validation accuracy:",
    max(history.history["val_accuracy"])
)

print(
    "Model saved as:",
    MODEL_PATH
)

print("========================================")
print("FINGERPRINT-X V2 READY")
print("========================================")