"""
Training utilities for portfolio optimization model.
"""

import os
import logging
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from model.model import SharpeLoss


def train_model(config, model, train_loader, val_loader):
    """
    Train the portfolio optimization model.
    
    Args:
        config (dict): Configuration dictionary
        model (nn.Module): Portfolio model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    curr_lr = float(config["training"]["lr"])
    optimizer = optim.Adam(model.parameters(), lr=curr_lr)
    criterion = SharpeLoss()

    best_val_sharpe = -float("inf")
    n_epochs = config["training"]["n_epochs"]
    steps_per_epoch = config["training"].get("steps_per_epoch", 12)

    for epoch in range(n_epochs):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{n_epochs}")
        for i, (x, y, masks) in enumerate(pbar):
            if i >= steps_per_epoch:
                break

            x, y, masks = x.to(device), y.to(device), masks.to(device)

            optimizer.zero_grad()
            weights, _ = model(x, masks)
            loss = criterion(weights, y, masks)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({"Sharpe": -np.mean(train_losses)})

        # Validation
        val_sharpe = validate(model, val_loader, criterion, device)
        logging.info(f"Epoch {epoch+1} | Val Sharpe: {val_sharpe:.4f}")

        # Save best model
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            os.makedirs(config["paths"]["model_save_dir"], exist_ok=True)
            save_path = os.path.join(
                config["paths"]["model_save_dir"],
                f"best_model_cycle_{config['cycles']['cycle_idx']}.pt"
            )
            
            torch.save(model.state_dict(), save_path)
            logging.info(f"New best model saved with Sharpe: {val_sharpe:.4f}")


def validate(model, val_loader, criterion, device):
    """
    Validate the model on validation set.
    
    Args:
        model (nn.Module): Portfolio model
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to run validation on
        
    Returns:
        float: Validation Sharpe ratio
    """
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for x, y, masks in val_loader:
            x, y, masks = x.to(device), y.to(device), masks.to(device)

            weights, _ = model(x, masks)
            loss = criterion(weights, y, masks)
            val_losses.append(loss.item())
    
    return -np.mean(val_losses)
