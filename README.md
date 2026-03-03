# ☕ Coffee Roast Quality AI

**Deep Learning-powered Coffee Bean Roast Classification** — Validate roast consistency against production standards in real time.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Features

- **Multi-Architecture Benchmarking**: Trained and evaluated InceptionV3, ResNet-152 V2, and VGG-16 on identical splits
- **95% Accuracy**: InceptionV3 achieves state-of-the-art performance on the 4-class roast dataset
- **Real-Time QC Validation**: Upload a batch photo and instantly verify it against a target roast profile
- **Modular ML Pipeline**: Clean `src/` package separating ingestion, preprocessing, model engine, and inference
- **Dockerized**: Single `docker build` + `docker run` for fully reproducible deployments
- **Streamlit UI**: Intuitive interface for quality control operators — no ML knowledge required

---

## 📊 Model Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **InceptionV3** ⭐ | **0.95** | **0.97** | **0.97** | **0.97** |
| ResNet-152 V2 | 0.96 | 0.96 | 0.96 | 0.96 |
| VGG-16 | 0.91 | 0.92 | 0.91 | 0.91 |

---

## 🗂️ Dataset

- **Source**: [Kaggle — Coffee Bean Dataset (224×224)](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224)
- **Size**: 1,600 images across 4 classes
- **Classes**: `Dark` · `Green` · `Light` · `Medium`
- **Split**: 90% train / 10% validation

---

## 🧠 How It Works

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐    ┌──────────────┐
│  Upload Image│ -> │  Preprocessing  │ -> │  InceptionV3     │ -> │  QC Result   │
│  (JPG/PNG)   │    │  Resize 224×224 │    │  + Custom Head   │    │  PASS / FAIL │
└──────────────┘    └─────────────────┘    └──────────────────┘    └──────────────┘
```

### Architecture
Each model uses the same custom head on top of ImageNet-pretrained base:
```
BaseModel (frozen) → GlobalAveragePooling2D → Dense(1024, ReLU) → Dropout(0.5) → Dense(4, Sigmoid)
```
Trained with **Adam** optimizer, **Categorical Cross-Entropy** loss, for **25 epochs** with data augmentation (rotation, flip, zoom, shear).

---

## 🖥️ Demo

| Upload Batch Photo | Quality Verdict |
|---|---|
| Select target roast from sidebar | System predicts roast type and compares |
| ✅ **PASS** — batch matches the target profile | ❌ **FAIL** — roast mismatch with corrective guidance |

> Live demo: [coffeeroastclassification.streamlit.app](https://coffeeroastclassification.streamlit.app/)

---

## 📦 Quick Start

### Option 1: Docker (Recommended)
```bash
git clone https://github.com/<your-username>/CoffeeBeanRoastClassification.git
cd CoffeeBeanRoastClassification

docker build -t coffee-roast-ai .
docker run -p 7860:7860 coffee-roast-ai
```
Open `http://localhost:7860`

### Option 2: Local Python
```bash
git clone https://github.com/<your-username>/CoffeeBeanRoastClassification.git
cd CoffeeBeanRoastClassification

pip install -r requirements.txt
pip install -e .

streamlit run app.py
```

---

## 🗃️ Project Structure

```
├── app.py                  # Streamlit UI entry point
├── main.py                 # Training pipeline entry point
├── params.yaml             # All hyperparameters & config
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
├── setup.py                # Installable src package
├── config/
│   └── config.yaml
├── models/
│   └── inception_v1.hdf5   # Best trained model
├── src/
│   └── coffee_roast_ai/
│       ├── data_ingest.py      # Kaggle dataset download
│       ├── data_loader.py      # Augmented data generators
│       ├── preprocessing.py    # Inference image processing
│       ├── model_engine.py     # Build / train / load models
│       ├── utils.py            # Config reader
│       └── logger.py           # Logging setup
└── research/
    ├── CoffeebeansClassification-Inception.ipynb
    └── CoffeebeansClassification-Resnet.ipynb
```

---

## ⚙️ Configuration

All tunable parameters live in [`params.yaml`](params.yaml) — no code changes needed:

```yaml
data:
  image_size: [224, 224]
  batch_size: 64
  class_names: ['Dark', 'Green', 'Light', 'Medium']

model:
  base_model: "InceptionV3"
  learning_rate: 0.001
  dropout_rate: 0.5
  dense_units: 1024
  epochs: 25
```

---

## 🔧 Development

```bash
# Install in editable mode
pip install -e .

# Run training pipeline
python main.py

# Run app locally
streamlit run app.py

# Rebuild Docker image
docker build -t coffee-roast-ai .
```

---

## 🙏 Acknowledgments

- [Kaggle — gerry pio senka](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224) for the dataset
- [TensorFlow / Keras](https://www.tensorflow.org/) for model training infrastructure
- [Streamlit](https://streamlit.io/) for the interactive UI framework

---

**Made with ❤️ and a lot of ☕**

