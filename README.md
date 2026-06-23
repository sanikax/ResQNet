# 🚑 ResQNet — Ambulance Detection System

> CNN-based binary image classifier to detect ambulances (emergency vehicles) vs non-ambulances.  
> Built with TensorFlow/Keras | Mini-Project | SPPU Third Year Computer Engineering

---

## 📁 Project Structure

```
ResQNet/
├── dataset/
│   ├── train/
│   │   ├── ambulance/          ← Training ambulance images
│   │   └── no_ambulance/       ← Training regular vehicle images
│   └── test/
│       ├── ambulance/          ← Test ambulance images
│       └── no_ambulance/       ← Test regular vehicle images
│
├── models/
│   └── resqnet_best.h5         ← Saved best model (after training)
│
├── outputs/
│   ├── training_history.png    ← Accuracy & loss curves
│   ├── confusion_matrix.png    ← Evaluation matrix
│   ├── roc_curve.png           ← ROC-AUC curve
│   └── classification_report.txt
│
├── logs/
│   └── training_log.csv        ← Per-epoch metrics log
│
├── model.py        ← CNN architecture (ResQNet + MobileNetV2 option)
├── train.py        ← Data loading, augmentation, training loop
├── evaluate.py     ← Confusion matrix, classification report, ROC
├── predict.py      ← Single/batch image prediction
├── app.py          ← Streamlit web application
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone / download the project
cd ResQNet

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your dataset images to dataset/train/ and dataset/test/
```

---

## 🏃 How to Run

### Train the Model
```bash
python train.py
```
- Trains for up to 30 epochs with early stopping
- Saves best model to `models/resqnet_best.h5`
- Plots accuracy/loss curves to `outputs/`

### Evaluate the Model
```bash
python evaluate.py
```
- Generates confusion matrix, ROC curve, classification report

### Predict a Single Image (CLI)
```bash
python predict.py path/to/vehicle_image.jpg
```

### Run the Web App
```bash
streamlit run app.py
```
- Opens browser UI at `http://localhost:8501`
- Upload any vehicle image for live prediction

---

## 🧠 Model Architecture — ResQNet CNN

```
Input (128 × 128 × 3)
     │
     ▼
┌─────────────────────────────────┐
│  Conv Block 1: Conv2D(32) × 2   │  Low-level features: edges, textures
│  BatchNorm → MaxPool → Dropout  │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  Conv Block 2: Conv2D(64) × 2   │  Mid-level features: shapes, lights
│  BatchNorm → MaxPool → Dropout  │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  Conv Block 3: Conv2D(128) × 2  │  High-level: semantic features
│  BatchNorm → MaxPool → Dropout  │
└─────────────────────────────────┘
     │
     ▼
   Flatten
     │
Dense(256) → BatchNorm → Dropout(0.5)
     │
Dense(128) → Dropout(0.3)
     │
Dense(1, sigmoid)   →   [0 = Ambulance | 1 = Non-Ambulance]
```

**Optimizer:** Adam (lr=0.001)  
**Loss:** Binary Crossentropy  
**Metrics:** Accuracy, Precision, Recall, AUC  

---

## 🔄 Optional: Transfer Learning

Switch to MobileNetV2 in `train.py`:
```python
from model import build_mobilenet_model
model, base_model = build_mobilenet_model()
```
Recommended when dataset < 5000 images per class.

---

## 📊 Expected Results (indicative)

| Metric    | Custom CNN | MobileNetV2 |
|-----------|-----------|-------------|
| Accuracy  | ~88–92%   | ~93–97%     |
| Precision | ~87–91%   | ~92–96%     |
| AUC       | ~0.93     | ~0.97       |

---

## 👨‍💻 Author

**[Your Name]**  
Third Year B.E. — Computer Engineering  
Savitribai Phule Pune University (SPPU)
