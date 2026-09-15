# 🧬 OClS_Cancer_CNN

**OClS_Cancer_CNN** is a Python-based deep learning application that uses a **Convolutional Neural Network (CNN)** for cancer image classification and provides an interactive **Streamlit web interface**.

> ⚠️ **Disclaimer:** This project is intended for educational and research purposes only. It is not a medical diagnostic system and should not be used for clinical decisions.

---

## 🚀 Features

* 🧠 CNN-based cancer image classification
* 🖼️ Upload medical images through Streamlit
* 🔍 AI-powered image prediction
* 📊 Prediction confidence display
* ⚡ Interactive web interface
* 🐍 Built completely with Python
* 📱 Responsive Streamlit application
* 📈 Model-based image analysis

---

## 🛠️ Tech Stack

| Technology         | Purpose              |
| ------------------ | -------------------- |
| Python             | Programming language |
| TensorFlow / Keras | CNN model            |
| Streamlit          | Web application      |
| NumPy              | Numerical processing |
| OpenCV             | Image processing     |
| Pillow             | Image handling       |
| Matplotlib         | Visualization        |

---

## 🧠 How It Works

```text
        User
          │
          ▼
   Streamlit Web App
          │
          ▼
     Upload Image
          │
          ▼
   Image Preprocessing
          │
          ▼
      CNN Model
          │
          ▼
    Image Classification
          │
          ▼
 Prediction + Confidence
```

---

## 📂 Project Structure

```text
OClS_Cancer_cnn/
│
├── app.py
├── model/
│   └── cancer_cnn_model.h5
│
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── images/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/BSS2707/OClS_Cancer_cnn.git
```

### 2. Open the project

```bash
cd OClS_Cancer_cnn
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

Example:

```text
streamlit
tensorflow
numpy
opencv-python
pillow
matplotlib
scikit-learn
```

---

## ▶️ Run the Streamlit App

Start the application with:

```bash
streamlit run app.py
```

Streamlit will open the application in your browser.

Usually:

```text
http://localhost:8501
```

---

## 🖥️ Using the Application

1. Start the Streamlit application.
2. Upload a supported medical image.
3. The application preprocesses the image.
4. The CNN model analyzes the image.
5. The predicted class is displayed.
6. The application can display the model's confidence score.

Example:

```text
Upload Image
     ↓
[ Select Image ]
     ↓
Model Prediction
     ↓
Predicted Class
     ↓
Confidence
```

---

## 🧠 CNN Model

The CNN learns visual patterns from the training images.

Typical architecture:

```text
Input Image
     ↓
Convolution Layer
     ↓
Activation
     ↓
Pooling
     ↓
Convolution Layer
     ↓
Pooling
     ↓
Flatten
     ↓
Dense Layer
     ↓
Output Layer
```

The exact architecture depends on the model implemented in the project.

---

## 📊 Results

Add your actual model performance here:

| Metric              | Result |
| ------------------- | -----: |
| Training Accuracy   |      — |
| Validation Accuracy |      — |
| Test Accuracy       |      — |
| Precision           |      — |
| Recall              |      — |
| F1 Score            |      — |

Do not add results until they have been measured from your actual model.

---

## 📸 Application Preview

Add screenshots of your Streamlit application:

```markdown
![OClS Cancer CNN](images/app.png)
```

---

## 🔮 Future Improvements

* [ ] Improve CNN accuracy
* [ ] Add data augmentation
* [ ] Add confusion matrix
* [ ] Add Grad-CAM visualization
* [ ] Add multiple model support
* [ ] Improve Streamlit UI
* [ ] Add prediction history
* [ ] Deploy the application online
* [ ] Add model performance dashboard

---

## ⚠️ Medical Disclaimer

This application is a **student/educational AI project**.

CNN predictions can contain errors and should not be interpreted as a diagnosis. Medical decisions should always be made by qualified healthcare professionals using appropriate clinical evaluation.

---

## 👨‍💻 Author

**Bhavya S Solanki**

AI/ML Student & Developer

GitHub: **BSS2707**

Portfolio: **bhavyasolanki.online**

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐.

**Built with Python 🐍 + CNN 🧠 + Streamlit ⚡**
