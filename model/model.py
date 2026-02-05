"""
Neural network models for portfolio optimization.

Architecture:
    - SREM (Sequential Representation Extraction Module): Transformer-based feature extraction
    - CAAN (Cross-Asset Attention Network): Cross-asset relationship modeling
    - Portfolio Generator: Top-G long/short portfolio construction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Portfolio(nn.Module):
    """
    Transformer-based portfolio optimization model.
    
    The model processes sequential asset features through:
    1. SREM: Feature projection + positional encoding + transformer encoder
    2. CAAN: Cross-asset attention mechanism
    3. Portfolio Generator: Rank-based long/short portfolio construction
    
    Args:
        n_feats (int): Number of input features per asset
        lookback (int): Lookback window size
        d_model (int): Transformer model dimension
        n_head (int): Number of attention heads
        n_layers (int): Number of transformer encoder layers
        G (int): Number of assets to long/short
    """
    
    def __init__(self, n_feats, lookback, d_model=256, n_head=8, n_layers=4, G=20):
        super().__init__()
        self.G = G
        self.d_model = d_model
        self.lookback = lookback

        # SREM: Feature projection + transformer
        self.feature_proj = nn.Linear(n_feats, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, lookback, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_model*4, 
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # CAAN: Cross-asset attention
        self.caan_dim = d_model * lookback
        self.W_Q = nn.Linear(self.caan_dim, self.caan_dim, bias=False)
        self.W_K = nn.Linear(self.caan_dim, self.caan_dim, bias=False)
        self.W_V = nn.Linear(self.caan_dim, self.caan_dim, bias=False)
        self.score_net = nn.Sequential(nn.Linear(self.caan_dim, 1), nn.Tanh())

    def forward(self, x, masks=None):
        """
        Produce portfolio weights given input features.

        Args:
            x: (B, T, n_assets, lookback, n_feats) input features
            masks: (B, T, n_assets) mask for valid assets

        Returns:
            weights: (B, T, n_assets) portfolio weights
            scores: (B, T, n_assets) asset scores before portfolio construction
        """
        B, T, A, L, n_feats = x.shape
        
        # SREM (flatten B, T, A to process all asset-sequences in parallel)
        x = x.view(-1, L, n_feats)                   # (B*T*A, L, n_feats)
        x = self.feature_proj(x) + self.pos_encoder  # (B*T*A, L, d_model)
        r = self.transformer(x).view(B * T, A, -1)   # (B*T, A, d_model * L)

        # CAAN (attention across assets)
        Q, K, V = self.W_Q(r), self.W_K(r), self.W_V(r)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.caan_dim ** 0.5)
        if masks is not None:
            m = masks.view(B * T, 1, A)
            attn_scores = attn_scores.masked_fill(m == 0, float("-inf"))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        r_attended = torch.matmul(attn_weights, V)  # (B*T, A, caan_dim)
        
        scores = self.score_net(r_attended).squeeze(-1).view(B, T, A)  # (B, T, A)

        # Portfolio generator
        weights = self._generate_portfolio(scores, masks)

        return weights, scores

    def _generate_portfolio(self, scores, masks):
        """
        Generate long/short portfolio by selecting top-G and bottom-G assets.
        
        Args:
            scores: (B, T, A) asset ranking scores
            masks: (B, T, A) valid asset masks
            
        Returns:
            weights: (B, T, A) portfolio weights (positive for long, negative for short)
        """
        B, T, A = scores.shape
        scores = scores.view(-1, A)  # (B*T, A)

        if masks is not None:
            scores = scores.masked_fill(masks.view(-1, A) == 0, float("-inf"))

            temp_scores = -scores
            temp_scores = temp_scores.masked_fill(masks.view(-1, A) == 0, float("-inf"))
        
        # Select top G and bottom G assets
        top_val, top_idx = torch.topk(scores, self.G, dim=1)
        bot_val, bot_idx = torch.topk(temp_scores, self.G, dim=1)

        # Weigh top and bottom assets
        long_weights = F.softmax(top_val, dim=1)   # (B*T, G)
        short_weights = F.softmax(bot_val, dim=1)  # (B*T, G)

        # Assign weights
        weights = torch.zeros_like(scores)  # (B*T, A)
        weights.scatter_(1, top_idx, long_weights)
        weights.scatter_(1, bot_idx, -short_weights)

        weights = weights.view(B, T, A)
        return weights


class SharpeLoss(nn.Module):
    """
    Negative Sharpe ratio loss function.
    
    Optimizes portfolio to maximize risk-adjusted returns by minimizing
    the negative expected Sharpe ratio.
    
    Args:
        eps (float): Small constant for numerical stability
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, weights, fut_rets, masks=None):
        """
        Compute negative Sharpe ratio loss.
        
        Args:
            weights: (B, T, n_assets) portfolio weights
            fut_rets: (B, T, n_assets) future returns
            masks: (B, T, n_assets) mask for valid assets

        Returns:
            loss: negative Sharpe ratio (scalar)
        """
        if masks is None:
            masks = torch.ones_like(weights)
        
        # Portfolio returns
        port_rets = torch.sum(weights * fut_rets * masks, dim=-1)  # (B, T)

        mean_ret = torch.mean(port_rets, dim=1)  # (B,)
        std_ret = torch.std(port_rets, dim=1)    # (B,)

        sharpe = mean_ret / (std_ret + self.eps)  # (B,)
        return -torch.mean(sharpe)
