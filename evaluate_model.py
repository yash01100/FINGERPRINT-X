import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("================================")
print("FINGERPRINT-X MODEL EVALUATION")
print("================================")

IMG_SIZE = (128, 128)
TEST_DIR = "dataset/test"
MODEL_PATH = "fingerprint_gender_model.keras"

# Load model
print("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Model loaded")

images = []
labels = []

files = [
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith(".bmp")
]

print("Test images:", len(files))


def get_label(filename):

    parts = filename.split("__")[1]
    gender = parts.split("_")[0]

    if gender == "M":
        return 1

    if gender == "F":
        return 0

    return None


# Load test dataset
print("Loading test images...")

for filename in files:

    label = get_label(filename)

    if label is None:
        continue

    path = os.path.join(TEST_DIR, filename)

    image = tf.keras.utils.load_img(
        path,
        color_mode="grayscale",
        target_size=IMG_SIZE
    )

    image = tf.keras.utils.img_to_array(image)

    image = image / 255.0

    images.append(image)
    labels.append(label)


X_test = np.array(images, dtype=np.float32)
y_test = np.array(labels)


print("================================")
print("TEST DATA READY")
print("================================")

print("Images:", X_test.shape)
print("Labels:", y_test.shape)


# Prediction
print("================================")
print("RUNNING PREDICTIONS")
print("================================")

probabilities = model.predict(
    X_test,
    verbose=1
).flatten()

predictions = (
    probabilities >= 0.5
).astype(int)


# Accuracy
accuracy = accuracy_score(
    y_test,
    predictions
)


print("================================")
print("FINAL RESULTS")
print("================================")

print(
    "Test Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# Classification report
print("================================")
print("CLASSIFICATION REPORT")
print("================================")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Female",
            "Male"
        ]
    )
)


# Confusion matrix
print("================================")
print("CONFUSION MATRIX")
print("================================")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


print("================================")
print("✅ EVALUATION COMPLETE")
print("================================")