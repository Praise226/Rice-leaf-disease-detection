# Rice Leaf Disease Detection System

A mobile application for classifying rice leaf images into five categories, with Grad-CAM++ visual explanations. Developed as a final-year project at the University of Medical Sciences, Ondo (UNIMED).

> **Important:** This is an academic prototype for preliminary image-based screening. It is not a substitute for field diagnosis or advice from a qualified agronomist. Performance on the test dataset does not establish real-world field accuracy.

## Project overview

The system accepts a rice leaf image through a React Native mobile application, sends it to a FastAPI backend, and returns a predicted class, confidence score, and a Grad-CAM++ visualization.

### Supported classes

- Bacterial Blight
- Blast
- Brown Spot
- Healthy
- Tungro

## Model and evaluation

| Item | Details |
|---|---|
| Architecture | MobileNetV2 transfer learning |
| Pretrained weights | ImageNet |
| Input dimensions | 224 × 224 × 3 |
| Classification head | Global Average Pooling → Dense(128, ReLU) → Dropout(0.3) → Dense(5, Softmax) |
| Dataset split | 70% training / 10% validation / 20% testing |
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Test accuracy | **99.01%** |
| Explainability | Grad-CAM++ |

The reported 99.01% is the evaluation result for the model selected for this project. It is a dataset test result, not a claim of 99.01% accuracy on farms or under all field conditions. Results depend on the dataset, image quality, and acquisition conditions.

## System architecture

1. **Mobile application:** React Native, Expo, TypeScript, and Expo Router. Users select or capture a rice leaf image and view the prediction.
2. **API:** FastAPI receives the image, preprocesses it, runs inference, generates a Grad-CAM++ overlay, and returns JSON.
3. **Model:** TensorFlow/Keras MobileNetV2 classifier with five output classes.

## Repository structure

```text
Rice-leaf-disease-detection/
├── api/
│   └── main.py
├── model/
│   └── train.py
├── frontend/             # Expo frontend integration notes (app source pending)
├── requirements.txt
├── .gitignore
└── README.md
```

The frontend and demo folders are being organized for this repository. The trained model weights and dataset are not committed; provide your own model file and dataset paths when running the project.

## Run the API locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Provide the trained model

The API reads the model path from the `MODEL_PATH` environment variable. By default, it looks for `rice_model_api.h5` in the working directory.

Windows PowerShell:

```powershell
$env:MODEL_PATH="path/to/rice_model_api.h5"
```

macOS/Linux:

```bash
export MODEL_PATH="path/to/rice_model_api.h5"
```

### 3. Start the API

```bash
uvicorn api.main:app --host localhost --port 8001
```

For a phone running Expo Go to reach a backend running on your computer, configure the mobile app with the computer's reachable LAN address. The address is environment-specific and is not a public production endpoint.

Health check: `GET /ping`

Prediction endpoint: `POST /predict` with an image in the `file` form field.

## Train and evaluate

Arrange the dataset in class-named subfolders:

```text
data/rice-leaf-disease-images/
├── Bacterial Blight/
├── Blast/
├── Brown Spot/
├── Healthy/
└── Tungro/
```

Set `DATASET_DIR` and `OUTPUT_DIR`, then run:

```bash
python model/train.py
```

The script trains the classifier, evaluates the test split, saves model files, and generates evaluation outputs. Training from scratch may produce a different score; the reported 99.01% belongs to the selected evaluated model and should not be assumed for every rerun.

## Mobile application

The mobile client is built with **React Native + Expo + TypeScript** (not React.js). The cleaned Expo source package has been prepared separately; the `frontend/` directory currently contains integration notes and an environment example. The full app source still needs to be added to the repository.

Set the API base URL for your development environment before starting Expo. A local development API address is expected while the backend runs on your own computer; it is not a hosted public service.

## Limitations and responsible use

- The model classifies images based on patterns learned from its training data.
- Image quality, lighting, camera angle, leaf age, disease severity, and environmental conditions can affect predictions.
- A high confidence score is not proof that a prediction is correct.
- Grad-CAM++ is a visual explanation aid; highlighted regions do not independently prove a biological cause.
- Further testing with diverse field images and review by agricultural specialists are needed before operational deployment.

## Author

**Oluwagbogo Praise Olasehinde**  
B.Sc. Information Technology — University of Medical Sciences, Ondo (UNIMED)  
GitHub: [@Praise226](https://github.com/Praise226)
