"""
Evaluation and backtesting utilities for portfolio optimization model.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml

from data.data import create_data_loaders
from model.model import Portfolio


def calculate_metrics(returns):
    """
    Calculate annualized Sharpe ratio and maximum drawdown.
    
    Args:
        returns (np.array): Array of returns
        
    Returns:
        tuple: (annualized_sharpe, max_drawdown, cumulative_returns)
    """
    ann_sharpe = np.sqrt(12) * np.mean(returns) / (np.std(returns) + 1e-8)
    
    cum_rets = np.cumsum(returns)
    peak = np.maximum.accumulate(cum_rets)
    drawdown = peak - cum_rets
    max_dd = np.max(drawdown)
    
    return ann_sharpe, max_dd, cum_rets


def get_percentile(strategy_final_ret, mc_final_rets):
    """
    Calculate percentile rank of strategy return vs Monte Carlo simulations.
    
    Args:
        strategy_final_ret (float): Final return of the strategy
        mc_final_rets (np.array): Final returns of MC simulations
        
    Returns:
        float: Percentile rank (0-100)
    """
    count_below = np.sum(mc_final_rets < strategy_final_ret)
    percentile = (count_below / len(mc_final_rets)) * 100
    return percentile


def run_evaluation(config_path="configs/config.yaml"):
    """
    Run full evaluation pipeline including backtesting and Monte Carlo simulation.
    
    Args:
        config_path (str): Path to configuration file
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    G = config["model"]["G"]
    T = config["model"]["T"]
    
    # Load validation data
    data_path = config["paths"]["data_path"]
    data_loader = create_data_loaders(config, data_path)
    val_loader = data_loader["val"]
    val_ds = val_loader.dataset

    val_indices = np.arange(0, len(val_ds), T)

    # Load model
    model = Portfolio(
        n_feats=data_loader["n_feats"],
        lookback=config["model"]["lookback"],
        d_model=config["model"]["d_model"],
        n_head=config["model"]["n_head"],
        n_layers=config["model"]["n_layers"],
        G=G
    ).to(device)
    
    model_path = f"saved_models/best_model_cycle_{config['cycles']['cycle_idx']}.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Generate returns
    model_rets = []
    markowitz_rets = []
    
    print("Running Backtest...")
    with torch.no_grad():
        for i in val_indices:
            x, y, masks = val_ds[i]
            x, y, masks = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device), masks.unsqueeze(0).to(device)
            
            # Model Weights
            weights, _ = model(x, masks)
            step_rets = torch.sum(weights * y * masks, dim=-1).squeeze(0).cpu().numpy()
            model_rets.extend(step_rets)
            
            # Rolling Markowitz (simple rank-based mean-variance)
            for t in range(T):
                hist_feats = x[0, t]  # (n_assets, lookback, n_feats)
                mask_t = masks[0, t]  # (n_assets,)

                mu = torch.mean(hist_feats, dim=1)[:, 0]  # (n_assets,)
                mu = mu.masked_fill(mask_t == 0, float("-inf"))
                mu_short = -mu
                mu_short = mu_short.masked_fill(mask_t == 0, float("-inf"))

                m_top_val, m_top_idx = torch.topk(mu, G)
                m_bot_val, m_bot_idx = torch.topk(mu_short, G)

                m_weights = torch.zeros_like(mu)  # (n_assets,)
                m_weights.scatter_(0, m_top_idx, 1.0/G)
                m_weights.scatter_(0, m_bot_idx, -1.0/G)

                m_ret = (y[0, t, m_top_idx].mean() - y[0, t, m_bot_idx].mean()).item()
                markowitz_rets.append(m_ret)

    # MC simulation
    print(f"Generating {config['evaluation']['num_mc_simulations']:,} MC Paths...")
    num_sims = config["evaluation"]["num_mc_simulations"]
    all_y = val_ds.labels[val_indices].view(-1, val_ds.labels.shape[-1]).numpy()  # (n_episodes*T, n_assets)
    all_m = val_ds.masks[val_indices].view(-1, val_ds.masks.shape[-1]).numpy()  # (n_episodes*T, n_assets)
    
    N_steps, A = all_y.shape
    mc_results = np.zeros((num_sims, N_steps))

    for i in range(num_sims):
        daily_sim_rets = []
        for t in range(N_steps):
            valid_indices = np.where(all_m[t] == 1)[0]
            if len(valid_indices) >= 2*G:
                perm = np.random.permutation(valid_indices)
                long_idx = perm[:G]
                short_idx = perm[G:2*G]
                step_ret = all_y[t, long_idx].mean() - all_y[t, short_idx].mean()
                daily_sim_rets.append(step_ret)
            else:
                daily_sim_rets.append(0.0)
        
        mc_results[i] = np.cumsum(daily_sim_rets)

    # Calculate envelopes
    upper_95 = np.percentile(mc_results, 95, axis=0)
    lower_5 = np.percentile(mc_results, 5, axis=0)
    median_50 = np.percentile(mc_results, 50, axis=0)
    
    # Calculate metrics and plot
    model_sharpe, model_mdd, model_cum = calculate_metrics(np.array(model_rets))
    m_sharpe, m_mdd, m_cum = calculate_metrics(np.array(markowitz_rets))

    final_model_ret = model_cum[-1]
    final_markowitz_ret = m_cum[-1]
    final_mc_rets = mc_results[:, -1]

    model_perc = get_percentile(final_model_ret, final_mc_rets)
    m_perc = get_percentile(final_markowitz_ret, final_mc_rets)

    print("-" * 30)
    print(f"MODEL:")
    print(f"  Annualized Sharpe: {model_sharpe:.4f}")
    print(f"  Max DD: {model_mdd:.4f}")
    print(f"  Final Return: {final_model_ret:.4f}")
    print(f"  Percentile: {model_perc:.2f}%")
    print(f"MARKOWITZ:")
    print(f"  Annualized Sharpe: {m_sharpe:.4f}")
    print(f"  Max DD: {m_mdd:.4f}")
    print(f"  Final Return: {final_markowitz_ret:.4f}")
    print(f"  Percentile: {m_perc:.2f}%")
    print("-" * 30)

    plt.figure(figsize=(12, 7))
    time_axis = np.arange(len(model_cum))
    
    # Shaded envelope
    plt.fill_between(time_axis, lower_5, upper_95, color="gray", alpha=0.2, label="MC 5th-95th Percentile")
    plt.plot(time_axis, median_50, color="gray", linestyle="--", alpha=0.5, label="MC Median")

    # Markowitz performance
    plt.plot(time_axis, m_cum, color="green", linewidth=2, label=f"Markowitz (Sharpe: {m_sharpe:.2f})")
    
    # Model performance
    plt.plot(time_axis, model_cum, color="blue", linewidth=2, label=f"AlphaPortfolio (Sharpe: {model_sharpe:.2f})")
    
    plt.title(f"Validation Backtest: {config['cycles']['val_start']} to {config['cycles']['val_end']}")
    plt.xlabel("Months")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(config["paths"]["img_dir"], exist_ok=True)
    plt.savefig(f"{config['paths']['img_dir']}/val_performance.png")
    plt.show()
    
    print(f"Evaluation Complete.")


if __name__ == "__main__":
    run_evaluation()
