"""
Rice Leaf Disease Detection — Model Training Pipeline
======================================================
Architecture : MobileNetV2 (transfer learning, ImageNet weights)
Classes      : Bacterial Blight | Blast | Brown Spot | Healthy | Tungro
Split        : 70% train / 10% validation / 20% test
Explainability: Grad-CAM++ heatmaps

Usage
-----
Set the following environment variables before running:

    DATASET_DIR   — path to the dataset root folder
                    (sub-folders must be named after each class)
    OUTPUT_DIR    — path where model weights and plots are saved

Example (Windows):
    set DATASET_DIR=C:\your\path\Rice Leaf Disease Images
    set OUTPUT_DIR=C:\your\path\outputs
    python train.py

Example (Linux/macOS):
    export DATASET_DIR=/your/path/rice-leaf-disease-images
    export OUTPUT_DIR=/your/path/outputs
    python train.py
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ─── Paths (read from environment — no hardcoded personal paths) ──────────────
DATASET_DIR = os.environ.get("DATASET_DIR", "data/rice-leaf-disease-images")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR",  "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "mobilenetv2_final.keras")
API_MODEL_PATH  = os.path.join(OUTPUT_DIR, "rice_model_api.h5")

# ─── Hyperparameters ──────────────────────────────────────────────────────────
IMAGE_SIZE = 224
BATCH_SIZE = 32
CHANNELS   = 3
EPOCHS     = 20

CLASS_NAMES = [
    "Bacterial Blight",
    "Blast",
    "Brown Spot",
    "Healthy",
    "Tungro",
]

# ─── 1. Load dataset ──────────────────────────────────────────────────────────
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
)
print("Classes found:", dataset.class_names)

# ─── 2. Train / validation / test split (70 / 10 / 20) ───────────────────────
def get_dataset_partitions_tf(
    ds,
    train_split=0.7,
    val_split=0.1,
    test_split=0.2,
    shuffle=True,
    shuffle_size=10000,
):
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size   = int(val_split   * ds_size)

    train_ds = ds.take(train_size)
    val_ds   = ds.skip(train_size).take(val_size)
    test_ds  = ds.skip(train_size + val_size)

    return train_ds, val_ds, test_ds


train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

print(f"Training batches  : {len(train_ds)}")
print(f"Validation batches: {len(val_ds)}")
print(f"Test batches      : {len(test_ds)}")

# ─── 3. Performance pipeline ──────────────────────────────────────────────────
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds  = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# ─── 4. Preprocessing layers ──────────────────────────────────────────────────
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0 / 255),
])

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
])

# ─── 5. Model architecture ────────────────────────────────────────────────────
# MobileNetV2 pretrained on ImageNet; top layers frozen during initial training
base_model = MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # freeze feature extractor

# Custom classification head
inputs  = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
x       = resize_and_rescale(inputs)
x       = data_augmentation(x)
x       = base_model(x, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(128, activation="relu")(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

model = Model(inputs, outputs)
print(f"Total parameters: {model.count_params():,}")

# ─── 6. Compile ───────────────────────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

# ─── 7. Callbacks ─────────────────────────────────────────────────────────────
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
)

# ─── 8. Training ──────────────────────────────────────────────────────────────
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds,
    callbacks=[early_stopping],
)

# ─── 9. Evaluate on test set ──────────────────────────────────────────────────
scores = model.evaluate(test_ds)
print(f"Test Loss    : {scores[0]:.4f}")
print(f"Test Accuracy: {scores[1] * 100:.2f}%")

# ─── 10. Save model ───────────────────────────────────────────────────────────
model.save(MODEL_SAVE_PATH)
print(f"Model saved → {MODEL_SAVE_PATH}")

# Export as .h5 for FastAPI inference
model.save(API_MODEL_PATH)
print(f"API model saved → {API_MODEL_PATH}")

# ─── 11. Training curves ──────────────────────────────────────────────────────
acc     = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss    = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(acc,     label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(loss,     label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Training curves saved")

# ─── 12. Classification report ────────────────────────────────────────────────
y_pred, y_true = [], []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

report_df = pd.DataFrame(
    classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
).transpose()
report_df.to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"))
print("Classification report saved")

# ─── 13. Confusion matrix ─────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Confusion matrix saved")

# ─── 14. Grad-CAM++ implementation ───────────────────────────────────────────
def get_gradcam_plusplus(model, img_array, layer_name, class_idx):
    """
    Generates a Grad-CAM++ saliency map for the given class index.

    Parameters
    ----------
    model      : trained Keras model
    img_array  : preprocessed image batch (1, H, W, C)
    layer_name : name of the target convolutional layer (e.g. 'Conv_1')
    class_idx  : index of the predicted class

    Returns
    -------
    cam : normalised heatmap array (H, W) in range [0, 1]
    """
    base_model = model.get_layer("mobilenetv2_1.00_224")

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer(layer_name).output,
            base_model.output,
        ],
    )

    with tf.GradientTape() as tape:
        inputs       = tf.cast(img_array, tf.float32)
        preprocessed = model.layers[1](inputs)   # resize_and_rescale
        preprocessed = model.layers[2](preprocessed)  # data_augmentation

        conv_outputs, base_predictions = grad_model(preprocessed)
        loss = base_predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    grads_power_2 = grads ** 2
    grads_power_3 = grads ** 3

    sum_activations = tf.reduce_sum(conv_outputs, axis=(1, 2))

    eps         = 1e-7
    alpha_num   = grads_power_2
    alpha_denom = (
        2.0 * grads_power_2
        + sum_activations[:, tf.newaxis, tf.newaxis, :] * grads_power_3
        + eps
    )
    alphas    = alpha_num / alpha_denom
    relu_grads = tf.nn.relu(grads)
    weights    = tf.reduce_sum(alphas * relu_grads, axis=(1, 2))

    conv_outputs = conv_outputs[0]
    weights      = weights[0]

    cam = tf.reduce_sum(weights * conv_outputs, axis=-1)
    cam = tf.nn.relu(cam)
    cam = cam.numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)

    return cam


def overlay_heatmap(img_array, cam, alpha=0.4):
    """Overlays the Grad-CAM++ heatmap on the original image."""
    heatmap = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlaid = cv2.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)
    return heatmap, overlaid


# ─── 15. Grad-CAM++ — single sample ──────────────────────────────────────────
for images_batch, labels_batch in test_ds.take(1):
    img        = images_batch[0].numpy().astype("uint8")
    true_label = CLASS_NAMES[labels_batch[0].numpy()]
    img_array  = tf.expand_dims(images_batch[0], axis=0)

    preds          = model.predict(img_array, verbose=0)
    pred_class_idx = np.argmax(preds[0])
    pred_label     = CLASS_NAMES[pred_class_idx]
    confidence     = preds[0][pred_class_idx] * 100

    cam, (heatmap, overlaid) = get_gradcam_plusplus(model, img_array, "Conv_1", pred_class_idx), \
                                overlay_heatmap(img, get_gradcam_plusplus(model, img_array, "Conv_1", pred_class_idx))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(img);      plt.title(f"Original\nTrue: {true_label}");              plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(heatmap);  plt.title("Grad-CAM++ Heatmap");                         plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(overlaid); plt.title(f"Overlay\nPred: {pred_label} ({confidence:.1f}%)"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gradcam_test.png"), dpi=300, bbox_inches="tight")
    plt.show()
    print(f"True: {true_label} | Predicted: {pred_label} | Confidence: {confidence:.1f}%")
    break

# ─── 16. Grad-CAM++ — one sample per class ───────────────────────────────────
plt.figure(figsize=(15, 12))
found_classes = {}

for images_batch, labels_batch in test_ds:
    for i in range(len(images_batch)):
        label_idx  = labels_batch[i].numpy()
        label_name = CLASS_NAMES[label_idx]

        if label_name not in found_classes and len(found_classes) < len(CLASS_NAMES):
            img        = images_batch[i].numpy().astype("uint8")
            img_array  = tf.expand_dims(images_batch[i], axis=0)
            preds      = model.predict(img_array, verbose=0)
            pred_idx   = np.argmax(preds[0])
            pred_label = CLASS_NAMES[pred_idx]
            confidence = preds[0][pred_idx] * 100

            cam              = get_gradcam_plusplus(model, img_array, "Conv_1", pred_idx)
            heatmap, overlaid = overlay_heatmap(img, cam)
            found_classes[label_name] = (img, heatmap, overlaid, pred_label, confidence)

    if len(found_classes) == len(CLASS_NAMES):
        break

for idx, (true_label, (img, heatmap, overlaid, pred_label, confidence)) in enumerate(found_classes.items()):
    plt.subplot(5, 3, idx * 3 + 1); plt.imshow(img);      plt.title(f"Original\nTrue: {true_label}", fontsize=8);               plt.axis("off")
    plt.subplot(5, 3, idx * 3 + 2); plt.imshow(heatmap);  plt.title("Grad-CAM++ Heatmap", fontsize=8);                          plt.axis("off")
    plt.subplot(5, 3, idx * 3 + 3); plt.imshow(overlaid); plt.title(f"Overlay\nPred: {pred_label} ({confidence:.1f}%)", fontsize=8); plt.axis("off")

plt.suptitle("Grad-CAM++ Visualisation — All Disease Classes", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gradcam_all_classes.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Grad-CAM++ visualisation saved for all classes")
