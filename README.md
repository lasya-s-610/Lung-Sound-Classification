# Lung-Sound-Classification
## 📌 Objective

This project aims to classify lung sounds into various respiratory conditions using deep learning. By processing Mel spectrograms of lung sound recordings, the model supports early diagnosis of respiratory diseases.

---

## 🧠 Project Summary

- **Model:** RDLINet (Reduced-Dimension Lightweight Inception Network)
- **Accuracy:** **84%**
- **Platform:** MATLAB
- **Input:** 10-second Mel spectrograms of lung sound recordings
- **Classes:** 7 respiratory conditions

---

## 🧪 Data Preprocessing Pipeline

1. **Resampling** audio to 4 kHz
2. **Temporal snippet generation** into 10-second snippets  
3. **DFT-based baseline wander removal**
4. **Amplitude normalization**
5. **Mel spectrogram generation**
6. **Oversampling** for class balancing

---

## 🧩 Model Architecture – RDLINet

RDLINet is a lightweight inception-based CNN optimized for low-complexity environments.

Key features:
- Depthwise separable convolutions
- Parallel convolutional filters (Inception modules)
- Batch normalization
- Global average pooling
- Dense classification layer

Designed to strike a balance between **accuracy** and **computational efficiency**.

---

## 📈 Results

- **Validation Accuracy:** 84%
- **Conditions Classified:**  
  Asthma, Bronchiectasis, Bronchiolitis, COPD, Healthy, Pneumonia, URTI
- **Metrics Evaluated:** Accuracy, Precision, Recall, F1-Score

---
