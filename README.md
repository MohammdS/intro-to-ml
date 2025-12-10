# Machine Learning Algorithms From Scratch

A comprehensive collection of fundamental machine learning algorithms implemented purely in **Python** and **NumPy**. 


## 🧠 Implemented Models

### 1. Linear Regression (`linearReg.py`)
* **Closed-Form Solution:** Implemented using the Normal Equation $(X^T X)^{-1} X^T y$.
* **Robust Outlier Detection:** Features a dynamic cleaning mechanism using the **Interquartile Range (IQR)** method to automatically detect and remove anomalies.
* **3D Visualization:** Includes 3D plotting of the regression plane against data points.

### 2. Bayesian Classifiers (`GB_GNB_classifiers.py`)
* **Gaussian Naïve Bayes (GNB):** Assumes feature independence. Optimized for high-dimensional data.
* **Gaussian Bayes (GB):** Models feature covariance. Includes **regularization techniques** (epsilon addition) to handle singular covariance matrices.
* **Analysis:** Includes correlation matrix visualization and PCA-based decision boundary plotting.

### 3. Text Classification (`naive_bayes_text.py`)
* **Bag-of-Words Model:** A full NLP pipeline implementing Naïve Bayes for text categorization.
* **Log-Probability Optimization:** Implemented using log-sum-exp tricks to prevent numerical underflow in long documents.
* **Laplace Smoothing:** Handles unseen words in the test set robustly.

### 4. K-Nearest Neighbors (`knn_digits.py`)
* **Digit Recognition:** Applied to the sklearn Digits dataset.
* **Custom Distance Metrics:** Flexible implementation supporting:
    * **Euclidean:** Standard L2 distance.
    * **Manhattan:** L1 distance (robust to high dimensions).
    * **Cosine:** Angle-based distance (effective for image/text vectors).
    * **Mahalanobis:** Covariance-aware distance.

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn (used only for data loading, splitting, and PCA)

### Installation

```bash
git clone [https://github.com/YOUR_USERNAME/ml-algorithms-from-scratch.git](https://github.com/YOUR_USERNAME/ml-algorithms-from-scratch.git)
cd ml-algorithms-from-scratch
pip install numpy pandas matplotlib scikit-learn