import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

def plot_reconstruction(y_true, y_pred, save_path=None):
    """
    Generates the scatter plot for Gene Expression Reconstruction (Fig II-VII in report).
    """
    # Calculate metrics
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    pearson, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    spearman, _ = spearmanr(y_true.flatten(), y_pred.flatten())
    
    plt.figure(figsize=(8, 6))
    # Use a hexbin or small alpha for dense single-cell data
    plt.hexbin(y_true.flatten(), y_pred.flatten(), gridsize=50, cmap='Blues', mincnt=1)
    
    # Diagonal perfect fit line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f"Reconstruction: True vs Predicted Expression\n"
              f"R2={r2:.2f}, Pearson={pearson:.2f}, Spearman={spearman:.2f}")
    plt.xlabel("Normalized Gene Expression (True)")
    plt.ylabel("Reconstructed Expression (Predicted)")
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plots a normalized confusion matrix for cell-type annotation.
    Crucial for spotting errors in rare cell types.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (Normalized by True Label)")
    plt.ylabel("True Cell Type")
    plt.xlabel("Predicted Cell Type")
    plt.xticks(rotation=45, ha='right')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()