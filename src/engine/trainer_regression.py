import torch
import torch.nn as nn
import logging
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Any, Tuple

logger = logging.getLogger("CellSentence.RegressionTrainer")

class RegressionTrainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Linear Regression often benefits from SGD or Adam with specific LRs
        lr = float(config['reconstruction']['model']['learning_rate'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Report Section II.A.2: Fitted via least squares -> MSE Loss
        self.criterion = nn.MSELoss()

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            preds = self.model(x)
            loss = self.criterion(preds, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        """
        Calculates R2, Pearson, and Spearman as per report Table I.
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            preds = self.model(x)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
        # Concatenate batches
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)
        
        # 1. R-squared (Global or per gene)
        # Global R2 is usually reported for reconstruction tasks
        r2 = r2_score(y_true.flatten(), y_pred.flatten())
        
        # 2. Pearson (Correlation between predicted and actual expression vectors)
        # Report cites: "Pearson correlation of 0.815" [cite: 80]
        # We calculate per-cell correlation and average it
        pearsons = []
        for i in range(y_true.shape[0]):
            p, _ = pearsonr(y_true[i], y_pred[i])
            if not np.isnan(p): pearsons.append(p)
        avg_pearson = np.mean(pearsons)
        
        # 3. Spearman (Rank correlation)
        spearmans = []
        for i in range(y_true.shape[0]):
            s, _ = spearmanr(y_true[i], y_pred[i])
            if not np.isnan(s): spearmans.append(s)
        avg_spearman = np.mean(spearmans)
        
        return {
            "r2": r2,
            "pearson": avg_pearson,
            "spearman": avg_spearman
        }
    
    def fit(self, train_loader, test_loader):
        epochs = self.config['training']['epochs']
        logger.info(f"Training reconstruction for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            loss = self.train_one_epoch(train_loader)
            
            # Log every 10 epochs to reduce noise
            if epoch % 10 == 0 or epoch == epochs:
                metrics = self.evaluate(test_loader)
                logger.info(
                    f"Epoch {epoch:03d} | Loss (MSE): {loss:.4f} | "
                    f"R2: {metrics['r2']:.4f} | "
                    f"Pearson: {metrics['pearson']:.4f} | "
                    f"Spearman: {metrics['spearman']:.4f}"
                )