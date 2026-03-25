# 🌸 Iris Flower Classification

> **CodeAlpha Data Science Internship — Task 1**

A machine learning project to classify Iris flower species (**Setosa**, **Versicolor**, **Virginica**) based on sepal and petal measurements using Python and Scikit-learn.

---

## 📌 Project Overview

The Iris dataset is one of the most famous datasets in machine learning. This project demonstrates a complete ML pipeline — from data loading and exploratory analysis to model training and evaluation.

---

## 🗂️ Dataset

- **Source:** Built-in `sklearn.datasets.load_iris()`
- **Samples:** 150 (50 per class)
- **Features:** 4 numerical features
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
- **Target:** 3 species — Setosa, Versicolor, Virginica

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Computing |
| Matplotlib & Seaborn | Data Visualization |
| Scikit-learn | ML Model Building |

---

## 📊 Exploratory Data Analysis (EDA)

- Feature distribution histograms by species
- Pair plot for feature relationships
- Correlation heatmap

---

## 🤖 Models Used

| Model | CV Accuracy |
|-------|------------|
| K-Nearest Neighbors | ~96% |
| Support Vector Machine | ~97% |
| **Random Forest** ✅ | **~97%** |

> Random Forest selected as the best model based on cross-validation accuracy.

---

## 📈 Results

- **Test Accuracy:** ~97%
- All 3 species classified with high precision and recall
- Petal Length and Petal Width are the most important features

---

## 📁 File Structure

```
CodeAlpha_IrisFlowerClassification/
│
├── iris_classification.py   # Main Python script
├── README.md                # Project documentation
├── iris_feature_distributions.png
├── iris_pairplot.png
├── iris_correlation_heatmap.png
├── iris_confusion_matrix.png
└── iris_feature_importance.png
```

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/CodeAlpha_IrisFlowerClassification.git
cd CodeAlpha_IrisFlowerClassification

# Install dependencies
pip install scikit-learn pandas matplotlib seaborn numpy

# Run the script
python iris_classification.py
```

---

## 🙋 Author

**Your Name**
- LinkedIn: [your-linkedin](https://linkedin.com)
- GitHub: [your-github](https://github.com)

---

> Made with ❤️ during CodeAlpha Data Science Internship
