# 🍷 Wine Quality Prediction Analysis

## 🔍 Overview
This project focuses on predicting the quality of wine based on various physicochemical properties using machine learning techniques. It includes data preprocessing, visualization, feature engineering, handling imbalanced data with SMOTE, and building classification models to accurately predict wine quality.

---

## 📁 Dataset
The dataset used is `winequalityN.csv`, which contains information about red and white wine samples. It includes features like:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- **Quality** (target variable)

**Target variable:** `quality` (rated between 0 and 10)

---

wine-quality-prediction/
│
├── README.md
├── winequalityN.csv             # Your dataset
├── wine_quality_analysis.ipynb  # Jupyter Notebook with code
└── models/                      # Folder for trained models


## ✅ Key Steps

### 1. Data Loading & Preprocessing
- Read CSV using Pandas
- Handle missing values using `SimpleImputer`
- Convert categorical columns if any (e.g., wine type)
- Normalize or scale features if required

### 2. Exploratory Data Analysis (EDA)
- Box plots for outliers
- Correlation heatmap
- Distribution of quality scores

### 3. Handling Imbalanced Data
- Use **SMOTE** (`imblearn.over_sampling.SMOTE`) to oversample minority classes

### 4. Model Building
Models tried include:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost (optional)

Performance evaluated using:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

### 5. Model Evaluation
- Train-test split (80/20 or 70/30)
- Use `classification_report` and `confusion_matrix` from `sklearn`

---

## 📊 Sample Results

| Model              | Accuracy | F1-Score |
|-------------------|----------|----------|
| Logistic Regression | 0.68     | 0.66     |
| Random Forest       | 0.78     | 0.77     |
| Gradient Boosting   | 0.80     | 0.79     |

*Note: Results may vary based on data split and SMOTE settings.*

---

## 📦 Requirements

Install dependencies:

```bash
pip install pandas scikit-learn matplotlib seaborn imbalanced-learn
