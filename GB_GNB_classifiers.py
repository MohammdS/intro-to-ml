import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# Classifiers
# ==========================================
def classify_point_gaussian_bayes(train, test, y_train):
    """
    Gaussian Bayes Classifier (Full Covariance)
    """
    classes = np.unique(y_train)
    log_probs = np.zeros((test.shape[0], len(classes)))

    for i, c in enumerate(classes):
        X_c = train[y_train == c]
        mu = np.mean(X_c, axis=0)
        cov = np.cov(X_c, rowvar=False)
        
        # Calculate log determinant and inverse
        sign, log_det = np.linalg.slogdet(cov)
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Fallback for stability if singular
            cov += np.eye(cov.shape[0]) * 1e-4
            cov_inv = np.linalg.inv(cov)

        diff = test - mu
        # Mahalanobis distance
        mahalanobis = np.sum((diff @ cov_inv) * diff, axis=1)
        prior = np.log(len(X_c) / len(train))
        
        log_probs[:, i] = prior - 0.5 * log_det - 0.5 * mahalanobis

    return classes[np.argmax(log_probs, axis=1)]

def classify_point_gaussian_naive_bayes(train, test, y_train):
    """
    Gaussian Naive Bayes Classifier (Diagonal Covariance)
    """
    classes = np.unique(y_train)
    log_probs = np.zeros((test.shape[0], len(classes)))

    for i, c in enumerate(classes):
        X_c = train[y_train == c]
        mu = np.mean(X_c, axis=0)
        var = np.var(X_c, axis=0) + 1e-6  # Add epsilon
        prior = np.log(len(X_c) / len(train))

        term1 = -0.5 * np.sum(np.log(2 * np.pi * var))
        term2 = -0.5 * np.sum(((test - mu)**2) / var, axis=1)
        log_probs[:, i] = prior + term1 + term2

    return classes[np.argmax(log_probs, axis=1)]

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    print("--- Bayesian Decision Rule Analysis ---")
    
    # Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, 'data', 'GB_GNB_data.csv')

    print(f"Looking for file at: {filepath}")

    if not os.path.exists(filepath):
        print("Error: File not found.")
        print("Please ensure the data file is inside a 'data' folder next to this script.")
        exit()

    df = pd.read_csv(filepath)
    data = df.to_numpy()
    X, y = data[:, :-1], data[:, -1]
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data Loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Correlation Visualization
    corr_matrix = pd.DataFrame(X_train).corr()
    
    print("\nDisplaying Correlation Matrix...")
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    plt.title("Feature Correlation Matrix (Show the correlation between features)")
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.tight_layout()
    plt.show()
    
    # Determinant check (just for GNB info)
    det = np.linalg.det(np.cov(X_train, rowvar=False))
    print(f"Determinant of Covariance Matrix: {det}")

    # Run Models
    print("\nRunning Classifiers...")
    models = {
        'Gaussian Bayes': classify_point_gaussian_bayes, 
        'Naive Bayes': classify_point_gaussian_naive_bayes
    }
    
    results = {'Model': [], 'Train Acc': [], 'Test Acc': []}
    
    for name, func in models.items():
        train_pred = func(X_train, X_train, y_train)
        test_pred = func(X_train, X_test, y_train)
        
        results['Model'].append(name)
        results['Train Acc'].append(np.mean(train_pred == y_train))
        results['Test Acc'].append(np.mean(test_pred == y_test))

    results_df = pd.DataFrame(results)
    print(results_df)