"""
Main training script for portfolio optimization.

Usage:
    python main.py
"""

import logging
import yaml

from data.data import create_data_loaders
from model.model import Portfolio
from train.train import train_model


def main():
    """Main training pipeline."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    logging.info("Configuration loaded successfully")
    
    # Create data loaders
    data_path = config["paths"]["data_path"]
    data_loader = create_data_loaders(config, data_path)
    logging.info(f"Data loaders created. Number of features: {data_loader['n_feats']}")
    
    # Initialize model
    model = Portfolio(
        n_feats=data_loader["n_feats"],
        lookback=config["model"]["lookback"],
        d_model=config["model"]["d_model"],
        n_head=config["model"]["n_head"],
        n_layers=config["model"]["n_layers"],
        G=config["model"]["G"]
    )
    
    logging.info("Model initialized")
    logging.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Start training
    logging.info("Starting training...")
    train_model(config, model, data_loader["train"], data_loader["val"])
    
    logging.info("Training completed successfully!")


if __name__ == "__main__":
    main()
