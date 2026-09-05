from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2
import base64
import matplotlib.pyplot as plt

app = FastAPI()

# Allow React frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import os

MODEL_PATH = os.getenv("MODEL_PATH", "rice_model_api.h5")
MODEL = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully")

print("Model input:", MODEL.input_shape)
print("Model output:", MODEL.output_shape)
print("Model layers:")

for i, layer in enumerate(MODEL.layers):
    print(i, layer.name, layer.__class__.__name__)



CLASS_NAMES = ["Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Tungro"]
IMAGE_SIZE = 224

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(image)

def generate_gradcam_plusplus(model, img_array, layer_name, class_idx):
    base_model = model.get_layer('mobilenetv2_1.00_224')
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer(layer_name).output,
            base_model.output
        ]
    )
    input_tensor = tf.cast(img_array, tf.float32)
    preprocessed = model.layers[1](input_tensor)
    preprocessed = model.layers[2](preprocessed)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(preprocessed)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    grads_power_2 = grads ** 2
    grads_power_3 = grads ** 3
    sum_activations = tf.reduce_sum(conv_outputs, axis=(1, 2))
    eps = 1e-7
    alpha_num = grads_power_2
    alpha_denom = 2.0 * grads_power_2 + sum_activations[:, tf.newaxis, tf.newaxis, :] * grads_power_3 + eps
    alphas = alpha_num / alpha_denom
    relu_grads = tf.nn.relu(grads)
    weights = tf.reduce_sum(alphas * relu_grads, axis=(1, 2))
    conv_outputs = conv_outputs[0]
    weights = weights[0]
    cam = tf.reduce_sum(weights * conv_outputs, axis=-1)
    cam = tf.nn.relu(cam)
    cam = cam.numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)
    return cam

def overlay_heatmap_on_image(img_array, cam, alpha=0.4):
    heatmap = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlaid = cv2.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)
    return overlaid

def image_to_base64(img_array):
    img = Image.fromarray(img_array.astype(np.uint8))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.get("/ping")
async def ping():
    return {"message": "Rice Disease Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image
    file_data = await file.read()

    print("\n==============================")
    print("NEW IMAGE RECEIVED")
    print("Filename:", file.filename)
    print("File size:", len(file_data), "bytes")

    # Read and preprocess image
    image = read_file_as_image(file_data)

    print("Processed image shape:", image.shape)
    print("Pixel range:", image.min(), "to", image.max())

    img_batch = np.expand_dims(image, axis=0)

    # Get prediction
    predictions = MODEL.predict(img_batch, verbose=0)

    print("\nMODEL PREDICTIONS:")
    for name, probability in zip(CLASS_NAMES, predictions[0]):
        print(f"{name}: {probability * 100:.4f}%")

    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = float(predictions[0][predicted_class_idx]) * 100

    print("\nFINAL PREDICTION:")
    print("Class:", predicted_class)
    print("Confidence:", round(confidence, 2), "%")
    print("==============================\n")

    # Generate Grad-CAM++ heatmap
    cam = generate_gradcam_plusplus(
        MODEL,
        img_batch,
        "Conv_1",
        predicted_class_idx
    )

    overlaid = overlay_heatmap_on_image(image, cam)
    heatmap_base64 = image_to_base64(overlaid)

    status = (
        "Healthy Leaf"
        if predicted_class == "Healthy"
        else "Disease Detected"
    )

    return {
        "status": status,
        "class": predicted_class,
        "confidence": round(confidence, 2),
        "heatmap": heatmap_base64
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)
