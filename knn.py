import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# Distance Metrics
# ==========================================
def Euclidean(test, data):
    """
    Calculates L2 distance (straight line).
    """
    return np.sqrt(np.sum((test[:, np.newaxis, :] - data[np.newaxis, :, :])**2, axis=2))

def Manhattan(test, data):
    """
    Calculates L1 distance (sum of absolute differences).
    """
    return np.sum(np.abs(test[:, np.newaxis, :] - data[np.newaxis, :, :]), axis=2)

def Cosine(test, data):
    """
    Calculates Cosine distance (1 - Cosine Similarity).
    """
    eps = 1e-10 # Prevent division by zero
    # Normalize vectors to unit length
    test_norm = test / (np.linalg.norm(test, axis=1, keepdims=True) + eps)
    data_norm = data / (np.linalg.norm(data, axis=1, keepdims=True) + eps)
    
    # Cosine Distance = 1 - (A . B)
    return 1 - np.dot(test_norm, data_norm.T)

# ==========================================
# KNN Classifier
# ==========================================
def kNN_classify(data, labels, test, k, metric='Euclidean'):
    # Calculate Distances
    if metric == 'Euclidean':
        dists = Euclidean(test, data)
    elif metric == 'Manhattan':
        dists = Manhattan(test, data)
    elif metric == 'Cosine':
        dists = Cosine(test, data)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    predictions = []
    # Iterate over every test sample
    for i in range(test.shape[0]):
        # 1. Find indices of k nearest neighbors
        nearest_indices = np.argsort(dists[i])[:k]
        
        # 2. Get labels of neighbors
        nearest_labels = labels[nearest_indices]
        
        # 3. Majority Vote
        counts = np.bincount(nearest_labels)
        predicted_label = np.argmax(counts)
        predictions.append(predicted_label)

    return np.array(predictions)

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    print("--- KNN Digits Classification ---")
    
    # Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, 'data', 'knn_digits.csv')

    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        exit()

    df = pd.read_csv(filepath)
    X = df.drop('label', axis=1).values
    y = df['label'].values

    # Split Data
    # We split first to ensure we evaluate on unseen data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=21, stratify=y
    )
    
    # Run Evaluation & Log Results
    metrics = ['Manhattan', 'Euclidean', 'Cosine']
    k_values = [1, 3, 5, 7, 9]

    print("\n" + "="*40)
    print(f"{'Metric':<15} | {'k':<5} | {'Accuracy':<10}")
    print("="*40)

    for metric in metrics:
        for k in k_values:
            preds = kNN_classify(X_train, y_train, X_test, k, metric)
            acc = np.mean(preds == y_test)
            print(f"{metric:<15} | {k:<5} | {acc:.4f}")
        print("-" * 40)