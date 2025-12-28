"""
Ring Bond Predictor for the ALD architecture.

Predicts cyclic connections (ring bonds) between residues in the generated
peptide sequence. Operates concurrently with token generation.

Ring bond types:
    - 0: No bond
    - 1: R3-R3
    - 2: R1-R2
    - 3: R1-R3
    - 4: R3-R2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple

from ..core.attention import MultiHeadAttention
from ..core.layers import FeedForward


class RingBondPredictor(nn.Module):
    """
    Predicts ring bond connections between residues.
    
    Architecture:
        1. Receives context vectors from the Context Encoder
        2. Uses attention weights or explicit pair scoring
        3. Predicts bond type for each (i, j) pair where i < j
    
    Args:
        d_model: Model dimension (from context encoder)
        hidden_dim: Hidden dimension for prediction network
        num_bond_types: Number of bond types (default: 5 including no-bond)
        dropout: Dropout probability
    """
    
    BOND_TYPES = ['none', 'R3R3', 'R1R2', 'R1R3', 'R3R2']
    
    def __init__(
        self,
        d_model: int = 512,
        hidden_dim: int = 256,
        num_bond_types: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_bond_types = num_bond_types
        
        # Pair scoring network
        # Takes concatenated pair of context vectors
        self.pair_encoder = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Bond type classifier
        self.bond_classifier = nn.Linear(hidden_dim, num_bond_types)
        
        # Attention-based alternative scorer
        self.attention_projection = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_bond_types)
        )
        
    def forward(
        self,
        context_vectors: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict ring bonds from context vectors.
        
        Args:
            context_vectors: Context from encoder [batch_size, seq_len, d_model]
            attention_weights: Optional attention weights [batch_size, n_heads, seq_len, seq_len]
            mask: Sequence mask [batch_size, seq_len]
            
        Returns:
            Bond predictions [batch_size, num_pairs, num_bond_types]
            where num_pairs = seq_len * (seq_len - 1) / 2 (upper triangular)
        """
        batch_size, seq_len, _ = context_vectors.shape
        device = context_vectors.device
        
        # Compute pairwise scores for upper triangular pairs (i < j)
        pair_features = []
        pair_indices = []
        
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                # Concatenate pair of context vectors
                pair_feat = torch.cat([
                    context_vectors[:, i, :],
                    context_vectors[:, j, :]
                ], dim=-1)  # [batch_size, d_model * 2]
                pair_features.append(pair_feat)
                pair_indices.append((i, j))
        
        if not pair_features:
            # Single token sequence - no pairs
            return torch.zeros(batch_size, 0, self.num_bond_types, device=device)
        
        # Stack pair features
        pair_features = torch.stack(pair_features, dim=1)  # [batch_size, num_pairs, d_model * 2]
        num_pairs = pair_features.size(1)
        
        # Encode pairs
        pair_features_flat = pair_features.view(-1, self.d_model * 2)
        pair_encoded = self.pair_encoder(pair_features_flat)
        pair_encoded = pair_encoded.view(batch_size, num_pairs, -1)
        
        # Classify bond types
        bond_logits = self.bond_classifier(pair_encoded)  # [batch_size, num_pairs, num_bond_types]
        
        # Optionally combine with attention-based scores
        if attention_weights is not None:
            # Average over heads
            avg_attn = attention_weights.mean(dim=1)  # [batch_size, seq_len, seq_len]
            
            # Extract upper triangular attention scores
            attn_scores = []
            for i, j in pair_indices:
                attn_scores.append(avg_attn[:, i, j])
            attn_scores = torch.stack(attn_scores, dim=1)  # [batch_size, num_pairs]
            
            # Project attention scores to bond logits
            attn_logits = self.attention_projection(attn_scores.unsqueeze(-1))
            
            # Combine
            bond_logits = bond_logits + 0.5 * attn_logits
        
        return bond_logits
    
    def predict_bonds(
        self,
        context_vectors: torch.Tensor,
        actual_length: int,
        threshold: float = 0.25,
        max_bonds: int = 1
    ) -> List[Dict]:
        """
        Predict ring bonds with confidence threshold.
        
        Args:
            context_vectors: [batch_size, seq_len, d_model]
            actual_length: Actual sequence length (excluding padding)
            threshold: Confidence threshold for bond prediction
            max_bonds: Maximum number of bonds to predict
            
        Returns:
            List of bond dictionaries with 'res1', 'res2', 'bond_type', 'confidence'
        """
        batch_size = context_vectors.size(0)
        device = context_vectors.device
        
        # Get bond logits
        bond_logits = self.forward(context_vectors[:, :actual_length, :])
        
        # Apply softmax to get probabilities
        bond_probs = F.softmax(bond_logits, dim=-1)  # [batch_size, num_pairs, num_bond_types]
        
        bonds_per_sample = []
        
        for b in range(batch_size):
            sample_probs = bond_probs[b]  # [num_pairs, num_bond_types]
            
            # Find bonds above threshold (excluding no-bond class 0)
            bonds = []
            
            # Get max probability and type for each pair
            max_probs, max_types = sample_probs.max(dim=-1)
            
            # Build pair indices
            pair_indices = []
            for i in range(actual_length):
                for j in range(i + 1, actual_length):
                    pair_indices.append((i, j))
            
            # Find bonds
            for pair_idx, ((i, j), prob, bond_type) in enumerate(zip(
                pair_indices, max_probs.tolist(), max_types.tolist()
            )):
                if bond_type > 0 and prob > threshold:  # Not no-bond and above threshold
                    bonds.append({
                        'res1': i + 1,  # 1-indexed
                        'res2': j + 1,
                        'bond_type': self.BOND_TYPES[bond_type],
                        'confidence': prob
                    })
            
            # Sort by confidence and take top bonds
            bonds = sorted(bonds, key=lambda x: -x['confidence'])[:max_bonds]
            bonds_per_sample.append(bonds)
        
        return bonds_per_sample if batch_size > 1 else bonds_per_sample[0]


class AutoregressiveRingPredictor(nn.Module):
    """
    Autoregressive ring bond predictor.
    
    Predicts whether the current token connects to any previous token,
    running at each generation step.
    
    Args:
        d_model: Model dimension
        hidden_dim: Hidden dimension
        num_bond_types: Number of bond types
    """
    
    BOND_TYPES = ['none', 'R3R3', 'R1R2', 'R1R3', 'R3R2']
    
    def __init__(
        self,
        d_model: int = 512,
        hidden_dim: int = 256,
        num_bond_types: int = 5
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_bond_types = num_bond_types
        
        # Query projection (current token)
        self.query_proj = nn.Linear(d_model, hidden_dim)
        
        # Key projection (previous tokens)
        self.key_proj = nn.Linear(d_model, hidden_dim)
        
        # Combined scoring
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_bond_types)
        )
        
    def forward(
        self,
        current_context: torch.Tensor,
        history_contexts: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict bonds between current token and all previous tokens.
        
        Args:
            current_context: Context for current token [batch_size, d_model]
            history_contexts: Contexts for previous tokens [batch_size, history_len, d_model]
            history_mask: Mask for history [batch_size, history_len]
            
        Returns:
            Bond logits [batch_size, history_len, num_bond_types]
        """
        batch_size = current_context.size(0)
        history_len = history_contexts.size(1)
        
        if history_len == 0:
            return torch.zeros(batch_size, 0, self.num_bond_types, device=current_context.device)
        
        # Project
        query = self.query_proj(current_context)  # [batch_size, hidden_dim]
        keys = self.key_proj(history_contexts)  # [batch_size, history_len, hidden_dim]
        
        # Expand query for each history position
        query_expanded = query.unsqueeze(1).expand(-1, history_len, -1)
        
        # Concatenate and score
        combined = torch.cat([query_expanded, keys], dim=-1)  # [batch_size, history_len, hidden_dim * 2]
        logits = self.scorer(combined)  # [batch_size, history_len, num_bond_types]
        
        # Apply mask
        if history_mask is not None:
            mask = history_mask.unsqueeze(-1)  # [batch_size, history_len, 1]
            logits = logits.masked_fill(~mask.bool(), float('-inf'))
        
        return logits
    
    def predict_connection(
        self,
        current_context: torch.Tensor,
        history_contexts: torch.Tensor,
        threshold: float = 0.5
    ) -> Optional[Dict]:
        """
        Predict if current token connects to any previous token.
        
        Args:
            current_context: [batch_size, d_model] (batch_size should be 1)
            history_contexts: [batch_size, history_len, d_model]
            threshold: Confidence threshold
            
        Returns:
            Bond dict or None if no bond predicted
        """
        logits = self.forward(current_context, history_contexts)
        probs = F.softmax(logits, dim=-1)  # [batch_size, history_len, num_bond_types]
        
        # Find max non-zero bond
        # Exclude class 0 (no bond)
        bond_probs = probs[0, :, 1:]  # [history_len, num_bond_types - 1]
        
        if bond_probs.numel() == 0:
            return None
        
        max_prob, max_idx = bond_probs.max(dim=1)  # [history_len]
        best_pos = max_prob.argmax().item()
        best_prob = max_prob[best_pos].item()
        best_type = max_idx[best_pos].item() + 1  # Add 1 because we excluded class 0
        
        if best_prob > threshold:
            return {
                'res1': best_pos + 1,  # Previous position (1-indexed)
                'bond_type': self.BOND_TYPES[best_type],
                'confidence': best_prob
            }
        
        return None
