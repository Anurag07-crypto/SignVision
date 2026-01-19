# 🤟 SignVision — Real-Time Sign Language Recognition

**SignVision** is a real-time **Sign Language Alphabet Recognition** system built using **PyTorch**, **OpenCV**, and a **MobileNetV2** deep learning model.  
It captures live video from a webcam, detects hand gestures within a defined region of interest (ROI), and predicts the corresponding sign language alphabet with confidence scores.

This project is lightweight, CPU-friendly, and designed for real-world demos and future scalability.

---

## ✨ Features

- 📸 Real-time webcam-based prediction
- 🧠 Deep learning model (MobileNetV2)
- 🎯 ROI-based hand detection for stable predictions
- 📊 Confidence score display
- 🖥️ Runs entirely on CPU
- 🔤 Supports **A–Z**, `Space`, and `Nothing` (28 classes)

---

## 🏗️ Model Overview

- **Base Model**: MobileNetV2 (pretrained on ImageNet)
- **Classifier**: Modified final layer for 28 sign language classes
- **Framework**: PyTorch
- **Inference**: Softmax-based confidence scoring

---

## 📂 Project Structure

```bash
signvision/
│
├── Models/
│   └── sign_language.pth          # Trained MobileNetV2 model weights
│
├── notebooks/
│   └── alphabet-using-sign-language.ipynb
│       # Training, experiments, and model development
│
├── venv/
│   # Virtual environment (ignored by git)
│
├── .gitignore
│
├── app.py
│   # Main application file for real-time sign language prediction
│
├── requirements.txt
│   # Project dependencies
│
└── README.md

git clone <(https://github.com/Anurag07-crypto/SignVision)>
cd signvision
