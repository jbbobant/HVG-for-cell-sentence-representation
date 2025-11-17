import torch
import torch.nn as nn
import logging
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from typing import Dict, Any, Tuple

logger = logging.getLogger("CellSentence.Trainer")

class Trainer:
    """
    Manages the training lifecycle of the model.
    """
    def __init__(
        self, 
        model: nn.Module, 
        config: Dict[str, Any], 
        device: torch.device
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer and Loss
        lr = float(config['model']['learning_rate'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
        # Early Stopping parameters
        self.patience = config['training']['patience']
        self.best_val_loss = float('inf')
        self.wait = 0
        
    def train_one_epoch(self, train_loader) -> float:
        """Runs one epoch of training."""
        self.model.train()
        total_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            
        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader) -> Tuple[float, float, float]:
        """
        Evaluates the model.
        Returns: (Loss, Accuracy, Macro-F1)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            
            logits = self.model(x)
            loss = self.criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
        avg_loss = total_loss / len(loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, acc, f1

    def fit(self, train_loader, val_loader) -> None:
        """
        Full training loop with Early Stopping.
        """
        epochs = self.config['training']['epochs']
        logger.info(f"Starting training for {epochs} epochs on {self.device}...")
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_acc, val_f1 = self.evaluate(val_loader)
            
            logger.info(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Val F1: {val_f1:.4f}"
            )
            
            # Early Stopping Check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.wait = 0
                # Save best model state (in memory or to disk)
                torch.save(self.model.state_dict(), "outputs/best_model.pt")
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Load best model for final state
        self.model.load_state_dict(torch.load("outputs/best_model.pt"))