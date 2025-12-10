# Intro to Machine Learning (From Scratch)

A collection of fundamental Machine Learning algorithms implemented using **pure Python**, **NumPy**, and **Pandas**.

The goal of this repository is to demonstrate a strong understanding of the mathematical foundations of ML by building models **without** relying on high-level frameworks like PyTorch or TensorFlow for the core logic. Scikit-Learn is used only for auxiliary tasks such as dataset splitting, PCA visualization, and metric evaluation.

## 🧠 Implemented Algorithms

### 1\. Linear Regression (`linearReg.py`)

A statistical approach to modeling the relationship between a scalar response and one or more explanatory variables using the **Closed-Form Solution** (Normal Equation).

**The Math:**
We minimize the Ordinary Least Squares (OLS) cost function by solving for weights $w$ directly:
$$w = (X^T X)^{-1} X^T y$$

**Key Techniques:**

  * **IQR Outlier Detection:** We use the Interquartile Range to filter noise.
      * $IQR = Q3 - Q1$
      * Bounds: $[Q1 - 1.5 \cdot IQR, \ Q3 + 1.5 \cdot IQR]$
  * **Evaluation:**
      * **MSE:** Mean Squared Error ($\frac{1}{n}\sum(y - \hat{y})^2$)
      * **$R^2$:** Coefficient of determination ($1 - \frac{SS_{res}}{SS_{tot}}$)

-----

### 2\. Bayesian Decision Rule (`GB_GNB_classifiers.py`)

Probabilistic classifiers based on applying **Bayes' Theorem**:
$$P(C|x) = \frac{P(x|C)P(C)}{P(x)}$$

**Included Models:**

  * **Gaussian Naïve Bayes (GNB):**
      * **Assumption:** Features are statistically independent.
      * **Math:** The Covariance Matrix $\Sigma$ is diagonal (off-diagonal elements are 0). This simplifies calculations but ignores feature interactions.
  * **Gaussian Bayes (GB):**
      * **Assumption:** Features may be correlated.
      * **Math:** Uses the **Full Covariance Matrix**. The likelihood is calculated using the multivariate Gaussian PDF:
        $$f(x) = \frac{1}{\sqrt{(2\pi)^k |\Sigma|}} \exp\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)\right)$$
      * **Regularization:** Adds a small epsilon ($\epsilon I$) to $\Sigma$ to prevent singular matrix errors during inversion.

-----

### 3\. Text Classification (`naive_bayes_text.py`)

A Natural Language Processing (NLP) pipeline using **Multinomial Naïve Bayes** on a Bag-of-Words representation.

**The Math:**

  * **Log-Space Calculation:** To prevent arithmetic underflow (multiplying many tiny probabilities results in 0), we sum logarithms instead:
    $$\hat{y} = \text{argmax} \left( \log P(C) + \sum \log P(w_i | C) \right)$$
  * **Laplace Smoothing:** Handles the "zero-frequency problem" (words in the test set that were never seen in the training set) by adding a small count $\alpha=1$:
    $$P(w|C) = \frac{\text{count}(w, C) + 1}{\text{count}(C) + |V|}$$

-----

### 4\. K-Nearest Neighbors (`knn.py`)

A non-parametric, lazy learning classifier applied to **8×8 handwritten digits** (64-dimensional feature space).

**The Logic:**
The model classifies a new data point $x$ by finding the $k$ training examples closest to it and taking a majority vote of their labels.

**Distance Metrics Implemented:**

  * **Euclidean ($L_2$):** Standard straight-line distance.
    $$d(x, y) = \sqrt{\sum (x_i - y_i)^2}$$
  * **Manhattan ($L_1$):** Sum of absolute differences (robust in high dimensions).
    $$d(x, y) = \sum |x_i - y_i|$$
  * **Cosine Similarity:** Measures the angle between vectors (ignores magnitude/intensity).
    $$d(x, y) = 1 - \frac{x \cdot y}{\|x\| \|y\|}$$

-----

## 🚀 Getting Started

### Prerequisites

  * Python 3.8+
  * NumPy
  * Pandas
  * Matplotlib
  * Scikit-Learn (for data splitting/PCA only)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/intro-to-ml.git
cd intro-to-ml
```

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## ▶️ Usage

Ensure your datasets are located in a `data/` folder at the project root.

```bash
# Linear Regression
python linearReg.py

# Bayesian Classifiers
python GB_GNB_classifiers.py

# Text Classification
python naive_bayes_text.py

# Digit Recognition (KNN)
python knn.py
```

## 🤝 Contributing

Pull requests, bug fixes, and issue submissions are welcome.

## 📝 License

This project is open-source under the **MIT License**.
