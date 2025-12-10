import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ==========================================
# Data Handling
# ==========================================
def get_data(filename='data/linearReg.csv'):

    # Load data
    if not os.path.exists(filename):
        print(f"File '{filename}' not found. Generating default data...")
        X_raw = np.array([
            [5.48, 7.15], [6.02, 5.44], [4.23, 6.45], [4.37, 10.91], [9.63, 3.83],
            [7.91, 15.28], [5.68, 12.25], [0.71, 0.87], [0.20, 11.32], [7.78, 8.70]
        ])
        y_raw = np.array([9.15, 11.87, 5.34, 8.91, 21.81, 17.23, 2.60, 4.27, -8.18, 11.41])
        data_stack = np.column_stack((X_raw, y_raw))
        pd.DataFrame(data_stack).to_csv(filename, index=False, header=False)
        
    df = pd.read_csv(filename)
    data = df.to_numpy()
    
    # Split into features (X) and target (y)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

# ==========================================
# Core Logic: Solvers & Processing
# ==========================================
def Linreg_sol(X, y):
    """
    Closed form solution for Linear Regression (Normal Equation).
    Returns weights [w1, w2, ..., bias]
    """
    # Add bias term (column of ones)
    ones = np.ones((X.shape[0], 1))
    X_aug = np.hstack((X, ones))
    
    # (X^T * X)^(-1) * X^T * y
    w = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y
    return w

def remove_outliers_iqr(X, y, multiplier=1.5):
    """
    Removes outliers from the dataset based on the IQR of the target variable y.
    
    Args:
        X (np.array): Feature matrix
        y (np.array): Target vector
        multiplier (float): IQR multiplier (standard is 1.5)
        
    Returns:
        X_clean, y_clean, mask: Cleaned data and the boolean mask used.
    """
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    mask = (y >= lower_bound) & (y <= upper_bound)
    
    X_clean = X[mask]
    y_clean = y[mask]
    
    return X_clean, y_clean, mask

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    print("--- Linear Regression Analysis ---")
    
    # Load Data
    try:
        X, y = get_data('linearReg.csv')
    except Exception as e:
        print(f"Error: {e}")
        exit()

    # Initial Model
    w = Linreg_sol(X, y)
    print(f'Linear line: y = {w[0]:.2f}*x1 + {w[1]:.2f}*x2 + {w[2]:.2f}')

    # Calculate MSE
    y_pred = X @ w[:2] + w[-1]
    mse = np.mean((y - y_pred)**2)
    print(f'Initial MSE: {mse:.2f}')

    # Visualization
    if X.shape[1] == 2:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        x1_range = np.linspace(X[:,0].min(), X[:,0].max(), 30)
        x2_range = np.linspace(X[:,1].min(), X[:,1].max(), 30)
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
        y_grid = w[0] * x1_grid + w[1] * x2_grid + w[2]
        
        ax.scatter(X[:,0], X[:,1], y, color='red', s=50, label='Original Data')
        ax.plot_surface(x1_grid, x2_grid, y_grid, color='blue', alpha=0.3)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Target')
        plt.title('Linear Regression Plane (Before Cleaning)')
        plt.show()

    # Outlier Removal
    print("\n--- Applying IQR Outlier Detection ---")
    X_clean, y_clean, mask = remove_outliers_iqr(X, y)
    
    outliers_removed = len(y) - len(y_clean)
    if outliers_removed > 0:
        print(f"Outliers removed: {outliers_removed}")
        
        # Refit model
        w_clean = Linreg_sol(X_clean, y_clean)
        
        # Recalculate MSE
        y_pred_clean = X_clean @ w_clean[:2] + w_clean[-1]
        mse_clean = np.mean((y_clean - y_pred_clean)**2)

        print(f'Linear line (after outlier removal): y = {w[0]:.2f}*x1 + {w[1]:.2f}*x2 + {w[2]:.2f}')
        print(f'Cleaned MSE: {mse_clean:.2f}')
    else:
        print("No outliers detected.")