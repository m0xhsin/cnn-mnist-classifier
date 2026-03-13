# cnn-mnist-classifier
A Convolutional Neural Network (CNN) for handwritten digit classification using the MNIST dataset. The project supports training and testing modes, with performance evaluation via a confusion matrix, and demonstrates core CNN concepts like convolution, pooling, batch normalization, and dropout.
# ✏️ CNN MNIST Classifier

![Python](https://img.shields.io/badge/Python-3-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2-orange)
![Keras](https://img.shields.io/badge/Keras-2-red)
![NumPy](https://img.shields.io/badge/NumPy-Data-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

**CNN MNIST Classifier** trains and tests a Convolutional Neural Network on the MNIST dataset of handwritten digits. It demonstrates key CNN concepts like convolution, pooling, batch normalization, dropout, and performance visualization via a confusion matrix.

---

## 🚀 Features
- 🖼️ Train CNN on MNIST handwritten digits  
- 🔁 Test model and visualize results with a confusion matrix  
- 💾 Save and load model weights (`model.weights.h5`)  
- 🧠 Multi-layer CNN architecture with Conv2D, MaxPooling, BatchNorm, Dropout, and Dense layers  

---

## 🛠️ Tech Stack
- Python 3  
- TensorFlow 2.x  
- Keras  
- NumPy  
- Matplotlib  
- scikit-learn  

---

## 🗂️ Dataset

The MNIST dataset (`mnist.npz`, 220 MB) is too large for GitHub. Download it manually:

[Download mnist.npz](https://drive.google.com/uc?export=download&id=1CDEniTt8rH43q91AuNjws2C2A8v0gbcX)

Save the file in the project folder before running `cnn.py`.

> **Optional:** Use `convert_data.py` to process custom image datasets into `.npz` format compatible with `cnn.py`.

---

## ▶️ How to Run

### Code:
```bash
Prepare / Convert Dataset (Optional):
python3 convert_data.py <image_dir> <H> <W> <C> <output.npz> <0|1>

Training the Model:
python3 cnn.py mnist.npz train

Testing the Model:
python3 cnn.py mnist.npz test
