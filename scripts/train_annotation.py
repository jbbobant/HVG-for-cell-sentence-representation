import sys
import os
import argparse
import logging
import torch
from pathlib import Path

# Ensure src is discoverable
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.common import load_config, setup_logger, seed_everything, get_device
from src.data import Preprocessor, create_dataloaders
from src.models import CellClassifier
from src.engine import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train Cell-Type Annotation Model")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the YAML configuration file (e.g., configs/base_config.yaml)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Override data path from config"
    )
    return parser.parse_args()

def main():
    # 1. Setup & Configuration
    args = parse_args()
    config = load_config(args.config)
    
    # Initialize Logging
    log_dir = config['paths']['output_dir']
    logger = setup_logger("CellSentence.Main", log_dir)
    
    logger.info("Configuration Loaded.")
    
    # Reproducibility
    seed_everything(config['project']['seed'])
    device = get_device(config['project']['device'])
    logger.info(f"Running on device: {device}")

    # 2. Data Pipeline
    logger.info("--- Starting Data Pipeline ---")
    
    # Determine input path (CLI argument takes precedence over config)
    input_path = args.data_path if args.data_path else config['paths']['input_data']
    
    # Load and Preprocess (QC, Filtering, Norm)
    preprocessor = Preprocessor(config['preprocessing'])
    preprocessor.load_data(input_path)
    
    # NOTE: If your input file is RAW, run processing:
    # preprocessor.process() 
    
    # NOTE: If your input file already has C2S embeddings, we use it directly.
    # If missing, you would insert the C2S embedding generation here.
    adata = preprocessor.adata

    # Create PyTorch DataLoaders
    train_loader, val_loader, test_loader, label_encoder = create_dataloaders(
        adata=adata,
        config=config,
        embedding_key="c2s_cell_embeddings", # Ensure this exists in your .h5ad
        label_key="cell_type"
    )
    
    # 3. Model Initialization
    logger.info("--- Initializing Model ---")
    
    # Dynamically determine input dimension from the data
    input_dim = adata.obsm["c2s_cell_embeddings"].shape[1]
    num_classes = len(label_encoder.classes_)
    
    logger.info(f"Input Dim: {input_dim} | Classes: {num_classes}")
    
    model = CellClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        config=config
    )
    
    # 4. Training Loop
    logger.info("--- Starting Training ---")
    trainer = Trainer(model, config, device)
    
    # Fit the model
    trainer.fit(train_loader, val_loader)
    
    # 5. Final Evaluation
    logger.info("--- Running Final Test Evaluation ---")
    # Load the best model saved during training
    model.load_state_dict(torch.load(os.path.join(log_dir, "best_model.pt")))
    model.to(device)
    
    test_loss, test_acc, test_f1 = trainer.evaluate(test_loader)
    
    logger.info("="*30)
    logger.info(f"FINAL TEST RESULTS")
    logger.info(f"Accuracy : {test_acc:.4f}")
    logger.info(f"Macro-F1 : {test_f1:.4f}")
    logger.info("="*30)

if __name__ == "__main__":
    main()