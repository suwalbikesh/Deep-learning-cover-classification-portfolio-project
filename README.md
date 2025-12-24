# 🌲 Forest Cover Type Classification with Deep Learning (TensorFlow)
## 📌 Project Overview

This project applies deep learning techniques to predict forest cover types using cartographic and environmental variables. The dataset is derived from US Geological Survey (USGS) and US Forest Service (USFS) data, representing minimally disturbed forest areas in northern Colorado.

The task is a multi-class classification problem where each land cell (30m × 30m) is classified into one of seven forest cover types using only terrain and soil features.

## 🎯 Objectives

- Build one or more deep learning classifiers
- Use TensorFlow with Keras
- Apply hyperparameter tuning to improve performance
- Evaluate models using robust classification metrics
- Create clean, modular, and reusable code
- Analyze misclassifications and suggest improvements

## 🌳 Forest Cover Types
1	Spruce / Fir
2	Lodgepole Pine
3	Ponderosa Pine
4	Cottonwood / Willow
5	Aspen
6	Douglas-fir
7	Krummholz

## 📊 Dataset Description

- Observations: 581,012
- Features: 54
  - 10 continuous cartographic variables
  - 4 binary wilderness area indicators
  - 40 binary soil type indicators
- Target: Cover_Type (integer values 1–7)

## Feature Examples

- Elevation (meters)
- Aspect (degrees azimuth)
- Slope (degrees)
- Distance to hydrology, roadways, fire points
- Hillshade indices
- Wilderness area & soil type (one-hot encoded)

## 🧠 Machine Learning Approach

- Problem Type: Multi-class classification
- Model: Fully connected neural network (MLP)
- Loss Function: Sparse categorical cross-entropy
- Optimizer: Adam
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score
- Regularization: Dropout + Early Stopping

## 🏗️ Project Structure
```
Deep-learning-cover-classification-portfolio-project/
│
├── data/
│   └── cover_data.csv
│
├── models/
│   └── forest_cover_model.h5
│
├── output/
│   ├── training_curves.png
│   └── confusion_matrix.png
│
├── src/
│   └── forest_cover_classification.py
│
└── README.md
```
## ⚙️ Installation & Setup
1️⃣ Clone the repository
```
git clone https://github.com/your-username/Deep-learning-cover-classification-portfolio-project.git
cd Deep-learning-cover-classification-portfolio-project
```

2️⃣ Create a virtual environment (optional but recommended)
```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```


## Key libraries used:

- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## ▶️ How to Run the Project
```
python src/Deep-learning-cover-classification-portfolio-project.py
```

### This will:

- Preprocess the dataset
- Train the deep learning model
- Tune hyperparameters with early stopping
- Evaluate performance on the test set
- Save the trained model and evaluation artifacts

## 📈 Results

- Test Accuracy: ~88–90%
- Best Performing Classes:
  - Spruce/Fir
  - Lodgepole Pine
- Most Challenging Classes:
  - Cottonwood/Willow
  - Aspen

## Outputs Generated

- Training accuracy & loss curves
- Confusion matrix heatmap
- Classification report (precision, recall, F1-score)
- Saved TensorFlow model (`.h5`)

## 🔍 Analysis & Insights

- Class imbalance impacts minority cover types
- Similar terrain characteristics cause overlap between some classes
- Deep learning models tend to favor majority classes due to gradient dominance

## 🧪 Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn

## 📌 Conclusion

This project demonstrates an end-to-end deep learning pipeline, from raw data preprocessing to model evaluation and interpretation. It highlights practical considerations such as class imbalance, hyperparameter tuning, and real-world dataset challenges.

It serves as a strong portfolio project showcasing applied deep learning skills using TensorFlow.

## 📬 Contact

If you have questions, suggestions, or would like to collaborate:

Author: Bikesh Suwal
GitHub: [https://github.com/suwalbikesh](https://github.com/suwalbikesh)
