# Rice Leaf Disease Detection System

A deep learning system that detects and classifies rice leaf diseases from images,
with visual explanations powered by Grad-CAM++.

Built as a final-year project at the University of Medical Sciences, Ondo (UNIMED).

---

## Overview

Rice diseases cause significant crop loss across farming communities. This system
allows farmers or agricultural workers to upload a photo of a rice leaf and instantly
receive a diagnosis — along with a heatmap showing exactly which part of the leaf
the model focused on.

---

## Detected Classes

| Disease | Description |
|---|---|
| Bacterial Blight | Caused by *Xanthomonas oryzae*, affects leaf margins |
| Blast | Fungal disease causing diamond-shaped lesions |
| Brown Spot | Fungal spots on leaves and grains |
| Healthy | No disease detected |
| Tungro | Viral disease transmitted by leafhoppers |

---

## Model Architecture

- **Base model:** MobileNetV2 (pretrained on ImageNet, frozen during training)
- **Custom head:** GlobalAveragePooling → Dense(128, ReLU) → Dropout(0.3) → Dense(5, Softmax)
- **Input size:** 224 × 224 × 3
- **Dataset split:** 70% train / 10% validation / 20% test
- **Explainability:** Grad-CAM++ heatmaps highlight disease-affected regions

### Results

| Metric | Score |
|---|---|
| Test Accuracy | 99.70% |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |

---

## Project Structure

Rice-leaf-disease-detection/
├── api/
│ └── main.py # FastAPI backend — handles image upload and prediction
├── model/
│ └── train.py # Full training pipeline with Grad-CAM++ implementation
├── demo.mp4 # Live demo of the mobile app interface
└── README.md


---

## How to Run the Backend

### 1. Install dependencies

```bash
pip install fastapi uvicorn tensorflow pillow opencv-python numpy matplotlib
```

### 2. Set environment variables

```bash
# Point to your trained model file
set MODEL_PATH=path/to/rice_model_api.h5        # Windows
export MODEL_PATH=path/to/rice_model_api.h5     # Linux/macOS
```

### 3. Start the server

```bash
uvicorn api.main:app --host localhost --port 8001
```

### 4. Test it

Visit `http://localhost:8001/ping` — you should see:
```json
{ "message": "Rice Disease Detection API is running" }
```

---

## How to Train the Model

### 1. Set environment variables

```bash
set DATASET_DIR=path/to/Rice Leaf Disease Images
set OUTPUT_DIR=path/to/outputs
```

### 2. Run training

```bash
python model/train.py
```

This will save the trained model, training curves, confusion matrix,
classification report, and Grad-CAM++ visualisations to your `OUTPUT_DIR`.

---

## Demo

See `demo.mp4` for a live walkthrough of the mobile app interface,
showing real-time disease detection and Grad-CAM++ heatmap overlay.

---

## Tech Stack

- **Model:** TensorFlow / Keras, MobileNetV2
- **Backend:** FastAPI, Uvicorn
- **Frontend:** React.js
- **Explainability:** Grad-CAM++ (OpenCV)
- **Evaluation:** scikit-learn, seaborn, matplotlib

---

## 👨‍💻 Author

**Olasehinde Oluwagbogo Oluwadunsin Praise**  
Final Year Student — Information Technology  
University of Medical Sciences, Ondo (UNIMED)  
GitHub: [@Praise226](https://github.com/Praise226)
