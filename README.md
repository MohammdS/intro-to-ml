# Intro to Machine Learning (From Scratch)

A collection of fundamental Machine Learning algorithms implemented using **pure Python**, **NumPy**, and **Pandas**.

The goal is to demonstrate a strong understanding of the mathematical foundations of ML by building models **without** high-level ML frameworks. Scikit-Learn is used only for helper tasks such as dataset splitting, PCA visualization, and metric evaluation.


## 🧠 Implemented Algorithms

### 1. Linear Regression (`linearReg.py`)

Implements linear regression using the **Closed-Form Solution**:

\[
w = (X^T X)^{-1} X^T y
\]

**Features:**
- IQR-based outlier detection and removal  
- MSE and R² performance evaluation  
- 3D visualization of the regression plane (before and after cleaning)

---

### 2. Bayesian Decision Rule (`GB_GNB_classifiers.py`)

Probabilistic classification using Bayes’ theorem.

**Included Models:**
- **Gaussian Naïve Bayes (GNB):** Assumes feature independence (diagonal covariance)  
- **Gaussian Bayes (GB):** Full covariance matrix to capture feature correlations  
  - Uses covariance inverse (Mahalanobis-like behavior)  
  - Includes epsilon regularization for singular matrices  

**Extras:**
- Feature correlation heatmaps for dataset interpretation

---

### 3. Text Classification (`naive_bayes_text.py`)

Bag-of-Words text classification using **Multinomial Naïve Bayes**.

**Techniques:**
- Log-probability computations to avoid underflow  
- Laplace smoothing to handle unseen words  
- Trains on a small news dataset (`text_clarification_train.csv`)

---

### 4. K-Nearest Neighbors (`knn.py`)

Non-parametric classifier applied to **8×8 handwritten digits**.

**Distance Metrics:**
- Euclidean  
- Manhattan  
- Cosine similarity  

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-Learn  

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/intro-to-ml.git
cd intro-to-ml
```

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## ▶️ Usage

Run any script:

```bash
# Linear Regression
python linear_regression.py

# Text Classification
python naive_bayes_text.py

# Digit Recognition (KNN)
python knn_digits.py
```

Make sure the `data/` folder exists at the project root.


## 🤝 Contributing

Pull requests and issue submissions are welcome.

---

## 📝 License

This project is open-source under the **MIT License**.
