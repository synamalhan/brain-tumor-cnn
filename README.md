# 🧠 Brain Tumor Classification and Segmentation

A deep learning project for binary classification and pixel-level segmentation of brain tumor MRI scans using CNNs (ResNet) and U-Net architectures. Includes data preprocessing, training scripts, and an interactive Streamlit app for visualization.

---

## 📁 Project Structure

```
BrainTumorCNN/
│
├── data/
│   ├── raw/                        # Original MRI images
│   ├── processed/                  # Preprocessed .npy arrays
│   └── labels.csv                  # CSV with metadata (image, class, shape, etc.)
│
├── notebooks/
│   ├── 1_data_preprocessing.ipynb
│   └── 2_model_testing.ipynb
│
├── models/
│   ├── resnet_classifier.py        # Binary classification model
│   └── unet_model.py               # U-Net segmentation model
│
├── scripts/
│   ├── train_classifier.py
│   └── train_unet.py
│
├── utils/
│   └── preprocessing.py            # Loading, resizing, normalizing images/masks
│
├── saved_models/
│   ├── resnet_model.h5
│   └── unet_model.h5
│
├── app/
│   └── streamlit_app.py            # Interactive demo app
│
├── requirements.txt
└── README.md
```

---

## 🚀 Features

- ResNet-based binary classifier for tumor detection  
- U-Net-based segmentation model for localizing tumor regions  
- Data normalization, augmentation, and binary mask thresholding  
- Evaluation with accuracy, F1-score, Dice, and IoU  
- Interactive **Streamlit App** to visualize predictions

---

## 🧪 Usage

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Train Classifier
```
python scripts/train_classifier.py
```

### 3. Train Segmentation Model
```
python scripts/train_unet.py
```

### 4. Run Streamlit App
```
streamlit run app/streamlit_app.py
```

---

## 📊 Metrics Tracked

- Accuracy, Precision, Recall, F1 (Classification)  
- Dice Coefficient, IoU (Segmentation)  
- Confusion Matrix  
- ROC AUC  
- Training curves saved as `training_curves.png`

---

## 📦 Dependencies

- Python 3.11  
- TensorFlow  
- NumPy, pandas, scikit-learn  
- Matplotlib, seaborn  
- Streamlit (for app UI)

---

## 🧠 Dataset

You can use public datasets like:

- [Brain MRI Segmentation Dataset (Kaggle)](https://www.kaggle.com/datasets/issamemm/brain-tumor-segmentation)  
- [Medical Segmentation Decathlon - Task 1 (BraTS)](http://medicaldecathlon.com/)

---

## ✨ Future Work

- Multi-class classification (e.g., glioma, meningioma, pituitary)  
- Multi-modal input (T1, T2, FLAIR)  
- Model ensembling  
- Better postprocessing for masks

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📜 License

[MIT License](LICENSE)
