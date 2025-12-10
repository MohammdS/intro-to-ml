# Intro to Machine Learning (From Scratch)

A collection of fundamental Machine Learning algorithms implemented using **pure Python**, **NumPy**, and **Pandas**.

This repository builds core ML models from the ground up without relying on high-level frameworks like PyTorch or TensorFlow for the core logic. Scikit-Learn is used only for auxiliary tasks such as data splitting and visualization.

## 🧠 Implemented Algorithms

### 1\. Linear Regression (`linearReg.py`)

A statistical model that predicts a target value by fitting the best straight plane through the data points.

  * **Technique:** Uses the **Closed-Form Solution** (Normal Equation) to calculate exact weights instantly, rather than iterative approximation.
  * **Outlier Detection:** Implements **Interquartile Range (IQR)** logic to automatically identify and remove data points that are statistically far from the norm, ensuring a more robust model.
  * **Evaluation:** Tracks Mean Squared Error (MSE) and R² scores to measure fit quality.

### 2\. Bayesian Decision Rule (`GB_GNB_classifiers.py`)

Probabilistic classifiers that predict the class with the highest probability given the input features.

  * **Gaussian Naïve Bayes (GNB):** Assumes all features are independent. It simplifies calculation by only looking at the variance of each feature individually.
  * **Gaussian Bayes (GB):** A more complex model that considers how features correlate with each other using the **Full Covariance Matrix**. It implicitly calculates distances based on the shape of the data distribution (Mahalanobis distance).
  * **Visualization:** Includes correlation matrices to visualize feature dependencies.

### 3\. Text Classification (`naive_bayes_text.py`)

A Natural Language Processing (NLP) tool that classifies news articles using the **Multinomial Naïve Bayes** algorithm.

  * **Bag-of-Words:** Converts text into numerical word counts.
  * **Log-Probabilities:** performs calculations in "log-space" to prevent numerical errors when handling very small probabilities associated with rare words.
  * **Laplace Smoothing:** Handles unseen words in the test data by ensuring no probability is ever exactly zero.

### 4\. K-Nearest Neighbors (`knn.py`)

A "lazy learning" classifier applied to **Handwritten Digit Recognition** (8x8 pixel images). It classifies new digits by comparing them to the database of known digits.

  * **Distance Metrics:**
      * **Euclidean:** Measures the straight-line distance between images (pixel intensity).
      * **Manhattan:** Measures distance along axes, often more robust in high dimensions.
      * **Cosine Similarity:** Measures the *angle* between image vectors, focusing on the shape pattern rather than pixel brightness.

## 🚀 Getting Started

### Prerequisites

  * Python 3.8+
  * NumPy
  * Pandas
  * Matplotlib
  * Scikit-Learn (for data tools only)

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
# Linear Regression Analysis
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
