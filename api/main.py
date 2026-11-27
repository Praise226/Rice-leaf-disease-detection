from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import base64
from io import BytesIO

import cv2
import numpy as np
import tensorflow as tf
import uvicorn
from PIL import Image, UnidentifiedImageError

app = FastAPI(title="Rice Leaf Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "rice_model_api.h5")
MODEL = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded successfully from: {MODEL_PATH}")

CLASS_NAMES = ["Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Tungro"]
IMAGE_SIZE = 224
TARGET_LAYER = "Conv_1"


def read_file_as_image(data: bytes) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.asarray(image, dtype=np.uint8)


def generate_gradcam_plusplus(model, img_array, layer_name, class_idx):
    """Generate Grad-CAM++ using the selected class score from the full classifier."""
    backbone = model.get_layer("mobilenetv2_1.00_224")
    conv_layer = backbone.get_layer(layer_name)
    feature_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[conv_layer.output, backbone.output],
    )

    # Preserve the model's own preprocessing and trained classification head.
    backbone_index = model.layers.index(backbone)
    classification_head = model.layers[backbone_index + 1:]
    inputs = tf.cast(img_array, tf.float32)
    preprocessed = model.layers[1](inputs, training=False)
    preprocessed = model.layers[2](preprocessed, training=False)

    with tf.GradientTape() as tape:
        conv_outputs, features = feature_model(preprocessed, training=False)
        predictions = features
        for layer in classification_head:
            predictions = layer(predictions, training=False)
        class_score = predictions[:, class_idx]

    grads = tape.gradient(class_score, conv_outputs)
    if grads is None:
        raise RuntimeError("Could not calculate Grad-CAM++ gradients for the selected class.")

    grads_power_2 = tf.square(grads)
    grads_power_3 = tf.pow(grads, 3)
    sum_activations = tf.reduce_sum(conv_outputs, axis=(1, 2))
    eps = 1e-7
    alpha_denom = (
        2.0 * grads_power_2
        + sum_activations[:, tf.newaxis, tf.newaxis, :] * grads_power_3
        + eps
    )
    alphas = grads_power_2 / alpha_denom
    weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(1, 2))

    cam = tf.reduce_sum(weights[0] * conv_outputs[0], axis=-1)
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + eps)
    return cam.numpy()

def overlay_heatmap_on_image(img_array: np.ndarray, cam: np.ndarray, alpha: float = 0.4):
    heatmap = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)


def image_to_base64(img_array: np.ndarray) -> str:
    image = Image.fromarray(img_array.astype(np.uint8))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.get("/ping")
async def ping():
    return {"message": "Rice Disease Detection API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")

    file_data = await file.read()
    if not file_data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        image = read_file_as_image(file_data)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    img_batch = np.expand_dims(image, axis=0)
    predictions = MODEL.predict(img_batch, verbose=0)
    predicted_class_idx = int(np.argmax(predictions[0]))
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = float(predictions[0][predicted_class_idx]) * 100

    try:
        cam = generate_gradcam_plusplus(MODEL, img_batch, TARGET_LAYER, predicted_class_idx)
        overlaid = overlay_heatmap_on_image(image, cam)
        heatmap_base64 = image_to_base64(overlaid)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not generate Grad-CAM++ visualization: {exc}")

    status = "Healthy Leaf" if predicted_class == "Healthy" else "Disease Detected"
    return {
        "status": status,
        "class": predicted_class,
        "confidence": round(confidence, 2),
        "heatmap": heatmap_base64,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)
