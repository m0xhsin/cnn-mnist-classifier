# cnn-mnist-classifier
A Convolutional Neural Network (CNN) for handwritten digit classification using the MNIST dataset. The project supports training and testing modes, with performance evaluation via a confusion matrix, and demonstrates core CNN concepts like convolution, pooling, batch normalization, and dropout.
# ✏️ CNN MNIST Classifier

![Python](https://img.shields.io/badge/Python-3-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2-orange)
![Keras](https://img.shields.io/badge/Keras-2-red)
![NumPy](https://img.shields.io/badge/NumPy-Data-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

A Convolutional Neural Network (CNN) project for **handwritten digit classification** using the **MNIST dataset**. It demonstrates **training and testing pipelines, performance visualization via confusion matrix**, and key CNN concepts like convolution, pooling, batch normalization, and dropout.

---

## 🚀 Features
- 🖼️ Train CNN on MNIST dataset  
- 🔁 Supports testing with confusion matrix visualization  
- 🧠 Multi-layer CNN with Conv2D, MaxPooling, BatchNorm, Dropout, and Dense layers  
- 💾 Save and load model weights (`model.weights.h5`) for reuse  
- 📊 Evaluate model accuracy quickly  

---

## 🛠️ Tech Stack
- Python 3  
- TensorFlow 2.x  
- Keras  
- NumPy  
- Matplotlib  
- scikit-learn  

---

## ▶️ How to Run

### 1. Training
```bash
python3 cnn.py mnist.npz train
