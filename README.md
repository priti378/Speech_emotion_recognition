# Speech_emotion_recognition
project
# Speech Emotion Classification using CNN

This project aims to recognize **emotions from speech/audio** using deep learning. It processes audio files, extracts meaningful features like MFCCs and ZCR, augments the data, trains a Convolutional Neural Network (CNN), and predicts the emotion expressed in a voice clip.

---

## Project Structure
emotion-classification/
├── data preprocess/ # Preprocessed CSV files
├── cnn_model_full.keras # Saved full Keras model
├── emotion_scaler.pkl # Saved StandardScaler
├── emotion_encoder.pkl # Saved OneHotEncoder
├── predict.py # Inference script
├── training.ipynb # Full training pipeline (notebook)
├── README.md # Project documentation

---

## 📝 1. Project Description

The aim is to design and implement a robust emotion classification system that works on speech data. The model learns patterns in human voice to accurately detect emotions, which can be useful in:

- Customer service sentiment detection
- AI assistants with emotion awareness
- Mental health monitoring
- Music or media analysis

## Emotions Supported

- 😐 Neutral  
- 😌 Calm  
- 😊 Happy  
- 😢 Sad  
- 😠 Angry  
- 😨 Fearful  
- 🤢 Disgust  
- 😲 Surprised  

---
## 🔄 2. Preprocessing Methodology

We used audio data from structured folders containing `.wav` files. Here’s how the preprocessing was done:

### 📁 Data Parsing:
- Extracted file paths and corresponding emotion labels using filename patterns.
- Created a combined DataFrame of paths and labels from speech and song folders.

### 🎵 Feature Extraction:
- Loaded audio files using `librosa` with:
  - `duration = 2.5s`
  - `offset = 0.6s`
- Extracted audio features:
  - **MFCC (Mel-Frequency Cepstral Coefficients)**
  - **ZCR (Zero Crossing Rate)**
  - **RMSE (Root Mean Square Energy)**
  
### 🎛️ Normalization:
- Used `StandardScaler` to normalize all extracted features.

### 🔁 Data Augmentation:
Each audio sample was augmented with:
- **Noise Injection**
- **Pitch Shifting**
- **Time Stretching**

### 🔁 Data Augmentation:
Each audio sample was augmented with:
- **Noise Injection**
- **Pitch Shifting**
- **Time Stretching**

This resulted in **4x data per file**, improving generalization.

---

## 🧠 3. Model Pipeline

The classifier is a **Convolutional Neural Network (CNN)** designed to work with 1D audio features.
### 🔨 Model Architecture:
Input → Conv1D(512) → BatchNorm → MaxPool →
Conv1D(512) → BatchNorm → MaxPool → Dropout →
Conv1D(256) → BatchNorm → MaxPool →
Conv1D(256) → BatchNorm → MaxPool → Dropout →
Conv1D(128) → BatchNorm → MaxPool → Dropout →
Flatten → Dense(512) → BatchNorm → Output(softmax)


### 🧪 Training Setup:
- Loss Function: `categorical_crossentropy`
- Optimizer: `Adam`
- Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`
- Epochs: 20 (with early stopping)
- Batch Size: 64
- Validation Split: 20%

---

## 📊 4. Accuracy Metrics

### ✅ Overall Performance on Test Data:
- **Accuracy**: `87.25%`
- **Macro F1 Score**: `0.87`
- **Weighted F1 Score**: `0.87`

### 📉 Per-Class Accuracy:
| Emotion    | Accuracy (%) |
|------------|--------------|
| Neutral    | 90.11        |
| Calm       | 88.40        |
| Happy      | 86.35        |
| Sad        | 88.87        |
| Angry      | 89.15        |
| Fearful    | 85.79        |
| Disgust    | 84.42        |
| Surprised  | 86.92        |

### 📌 Confusion Matrix:
A detailed confusion matrix is plotted showing true vs predicted labels for all 8 emotions.

---
---



## 🚀 How to Use

### ➤ Predict Emotion from Audio:
```bash
python predict.py path/to/audio.wav

output:
🎤 Emotion Detected: happy


## Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt


How to Train the Model
Open training.ipynb in Jupyter or Colab.

Follow all cells from preprocessing, feature extraction, model building, training, and evaluation.

The model and preprocessing tools will be saved as .keras and .pkl files.


🎯 How to Predict Emotion from Audio
Once your model is trained and saved!
use:
   predict.py



📊 Model Highlights
Uses MFCC, ZCR, and RMSE for feature extraction.

Includes data augmentation (noise, pitch, stretch).

CNN-based architecture for classification.

Achieves high accuracy and F1 score on custom test sets.

Includes confusion matrix and per-class accuracy reports.


✅ Project Goals
Build an end-to-end speech emotion recognition pipeline.

Support robust evaluation (confusion matrix, F1).

Make prediction reusable via predict.py.


📌 Credits
This project was built using open-source tools like librosa, TensorFlow, and scikit-learn.






 
