"""
Data loading and processing utilities for portfolio optimization.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PortfolioData(Dataset):
    """
    Dataset for portfolio optimization with multi-step episodes.
    
    Args:
        data_path (str): Path to CSV data file
        start_date (str): Start date for data filtering
        end_date (str): End date for data filtering
        T (int): Episode length (number of time steps)
        lookback (int): Lookback window for features
    """
    
    def __init__(self, data_path, start_date, end_date, T, lookback, **kwargs):
        self.T = T
        self.lookback = lookback
        
        df = pd.read_csv(data_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].sort_values(["date", "permno"])
        
        self.excl_cols = ["permno", "date", "rdq", "ret"]  # "ret" is the target
        self.feat_cols = [c for c in df.columns if c not in self.excl_cols]
        self.uniq_dates = np.sort(df["date"].unique())
        self.permnos = sorted(df["permno"].unique())
        self.permno_to_idx = {p: i for i, p in enumerate(self.permnos)}
        
        pivot_df = df.pivot(index="date", columns="permno", values=self.feat_cols + ["ret"])
        feature_data = pivot_df[self.feat_cols].values.reshape(
            len(self.uniq_dates), len(self.permnos), len(self.feat_cols)
        )  # (n_dates, n_assets, n_features)
        return_data = pivot_df["ret"].values  # (n_dates, n_assets)
        
        # Cross-sectional standardization
        mean = np.nanmean(feature_data, axis=1, keepdims=True)
        std = np.nanstd(feature_data, axis=1, keepdims=True)
        feature_data = (feature_data - mean) / (std + 1e-8)
        feature_data = np.nan_to_num(feature_data)
        
        self.sequences = []
        self.labels = []
        self.masks = []
        
        n_possible_steps = len(self.uniq_dates) - self.lookback
        for start_idx in range(0, n_possible_steps - self.T + 1):
            ep_seq = []    # later will be (T, n_assets, lookback, n_features)
            ep_label = []  # later will be (T, n_assets)
            ep_mask = []   # later will be (T, n_assets)
            
            for t in range(self.T):
                curr_t = start_idx + t
                window = feature_data[curr_t : curr_t + self.lookback]  # (lookback, n_assets, n_features)
                target_ret = return_data[curr_t + self.lookback]        # (n_assets,)
                
                # Mask out assets that are all NaNs in the target
                mask = ~np.isnan(target_ret)
                
                ep_seq.append(window.transpose(1, 0, 2))  # (n_assets, lookback, n_features)
                ep_label.append(np.nan_to_num(target_ret))
                ep_mask.append(mask.astype(float))
                
            self.sequences.append(np.stack(ep_seq))
            self.labels.append(np.stack(ep_label))
            self.masks.append(np.stack(ep_mask))

        # Convert to Tensors
        self.sequences = torch.tensor(np.array(self.sequences), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)
        self.masks = torch.tensor(np.array(self.masks), dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.masks[idx]


def create_data_loaders(config, data_path):
    """
    Create train and validation data loaders.
    
    Args:
        config (dict): Configuration dictionary
        data_path (str): Path to data file
        
    Returns:
        dict: Dictionary with 'train', 'val' loaders and 'n_feats'
    """
    T = config["model"]["T"]
    lookback = config["model"]["lookback"]
    batch_size = config["training"]["batch_size"]
    n_workers = config["training"]["n_workers"]

    train_ds = PortfolioData(
        data_path=data_path,
        start_date=config["cycles"]["train_start"],
        end_date=config["cycles"]["train_end"],
        T=T,
        lookback=lookback
    )

    train_data_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True
    )

    val_ds = PortfolioData(
        data_path=data_path,
        start_date=config["cycles"]["val_start"],
        end_date=config["cycles"]["val_end"],
        T=T,
        lookback=lookback
    )

    val_data_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True
    )

    return {
        "train": train_data_loader,
        "val": val_data_loader,
        "n_feats": len(train_ds.feat_cols)
    }
