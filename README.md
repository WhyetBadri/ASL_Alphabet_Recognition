# ASL Alphabet Recognition: High-Precision Hand Gesture Classification with MediaPipe and Random Forest

This project implements a robust, real-time American Sign Language (ASL) alphabet recognition system. By combining Google’s **MediaPipe** for high-fidelity hand landmarking with a **Random Forest Ensemble** for classification, the system achieves exceptional accuracy while remaining lightweight enough to run on standard consumer hardware without a dedicated GPU.

##  Significance & Impact

Communication barriers between the Deaf community and non-signers often necessitate expensive interpreters or slow text-based exchanges. This solution provides a bridge by translating 26 static hand gestures into text in real-time. Unlike traditional CNN-based approaches that process raw pixels—making them sensitive to lighting and background noise—this project extracts **geometric hand landmarks**, ensuring the model focuses exclusively on the structural pose of the hand.

##  System Architecture

The pipeline follows a modular design from data acquisition to real-time inference:

1. **Data Acquisition:** Custom-built webcam collector for personalized dataset generation.
2. **Feature Extraction:** MediaPipe AI maps 21 3D hand landmarks.
3. **Normalization Engine:** Transposition of coordinates to a relative local coordinate system.
4. **Inference Engine:** A 200-tree Random Forest classifier predicting the character with confidence scoring.

---

##  Methodology & Dataset

### 1. Dataset Generation

The dataset consists of **3,900 images** (150 samples per letter for all 26 ASL alphabet signs).

* **Source:** Captured via `webcam_collector.py` at 720p resolution.
* **Variety:** Collected under varying lighting conditions and slight hand rotations to ensure model generalization.

### 2. Feature Extraction & Preprocessing

Raw images are never fed into the classifier. Instead, we use a sophisticated preprocessing pipeline defined in `feature_extractor.py`:

* **Hand Landmarking:** MediaPipe detects 21 keypoints (wrists, joints, fingertips).
* **Relative Positioning:** Every coordinate  is recalculated relative to the wrist . This makes the model **translation-invariant** (you can sign anywhere in the frame).
* **Scaling/Normalization:** Coordinates are scaled by the maximum distance found in the hand cluster. This makes the model **scale-invariant** (it works regardless of how close your hand is to the camera).

---

##  Model Architecture & Regularization

### Random Forest Classifier

We utilized a **Random Forest** algorithm because of its inherent resistance to overfitting and its ability to handle multi-class classification with non-linear decision boundaries.

| Hyperparameter | Value | Reasoning |
| --- | --- | --- |
| **Estimators** | 200 | Balancing voting stability with computational latency. |
| **Criterion** | Gini | Optimal for categorical splitting of landmark features. |
| **Max Features** |  | Ensures diversity among trees to prevent overfitting. |
| **Random State** | 42 | Ensures reproducibility of the ensemble forest. |

### Regularization Strategy

To ensure the model generalizes to new users, we implemented several layers of regularization:

* **Feature Bagging:** Each tree in the forest sees a random subset of the 42 features (21 landmarks  2 coordinates), forcing the model to learn multiple ways to identify a letter (e.g., recognizing 'B' by both thumb position and finger extension).
* **Validation Split:** An 80/20 train-test split ensures we validate on 780 completely unseen gesture samples.
* **Early Stopping (Conceptual):** During development, we analyzed the OOB (Out-of-Bag) error to settle on 200 trees, preventing unnecessary complexity.

---

##  Performance Results

The model achieves a **Validation Accuracy of 98.5%**.

### Confusion Matrix Insights

Our error analysis showed that the most common (though rare) misclassifications occurred between:

* **'M' vs 'N':** Due to the subtle difference in thumb placement under the fingers.
* **'U' vs 'V':** Depending on the narrowness of the finger spread.

---

##  Installation & Reproducibility

### 1. Requirements

Ensure you are using **Python 3.11.9**. Install dependencies:

```bash
pip install -r requirements.txt

```

### 2. Step-by-Step Execution

1. **Collect Data:** Run `python webcam_collector.py` to capture your own gesture samples.
2. **Extract Features:** Run `python feature_extractor.py` to convert images into a normalized `.pickle` dataset.
3. **Train Model:** Run `python forest_trainer.py` to generate the `asl_forest_model.pkl` file.
4. **Live Recognition:** Run `python live_recognizer.py` to start the real-time webcam translator.

---

##  Project Structure

```text
├── asl_forest_model.pkl       # The trained Random Forest model
├── feature_extractor.py       # MediaPipe preprocessing pipeline
├── forest_trainer.py          # Scikit-learn training script
├── gesture_features.pickle    # Extracted numerical features
├── live_recognizer.py         # Real-time webcam inference app
├── requirements.txt           # Version-locked dependencies
└── webcam_collector.py        # Dataset acquisition tool

```
