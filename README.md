# Emotion Recognition Project

Real-time facial emotion recognition system built using Deep Learning, Convolutional Neural Networks (CNN), TensorFlow, and OpenCV.

## 📌 Overview

This project detects human faces through a webcam and predicts facial emotions in real time. It uses a trained CNN model to classify expressions into multiple emotion categories.

## 🎯 Features

* Real-time webcam emotion detection
* Face detection using Haar Cascade Classifier
* Deep Learning based emotion classification
* User-friendly and fast prediction
* Supports multiple emotions

## 😊 Emotion Classes

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy

## 📂 Project Structure

```text
EmotionRecognitionProject/
│── train.py
│── webcam.py
│── haarcascade_frontalface_default.xml
│── .gitignore
```

## 🚀 How to Run

### 1️⃣ Install Required Libraries

```bash
pip install tensorflow keras opencv-python numpy
```

### 2️⃣ Train the Model

```bash
python train.py
```

### 3️⃣ Run Real-Time Detection

```bash
python webcam.py
```

## 📸 Working Process

1. Webcam captures live video
2. Face is detected using OpenCV
3. Face image is resized and processed
4. CNN model predicts emotion
5. Emotion label displayed on screen

## 📈 Future Improvements

* Higher model accuracy
* Better UI dashboard
* Mobile app integration
* Attendance + Emotion analytics
* Stress detection system

## 👩‍💻 Author

Somya Khandelwal

## 📄 License

This project is for educational and learning purposes.
