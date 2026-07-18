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


"""
Method 2: Autoregressive Ring Bond Prediction
"""
class AutoregressiveRingPredictor(nn.Module):
    """
    Autoregressive ring bond predictor with R-group awareness.
    
    Predicts whether the current token connects to any previous token,
    running at each generation step. Uses both context vectors and
    R-group embeddings for prediction.
    
    Architecture:
        - Position head: predicts bonding score for each history position
        - Type head: predicts bond type (4 classes: R3R3, R1R2, R1R3, R3R2)
    
    Args:
        d_model: Context encoder output dimension
        r_dim: R-group embedding dimension (typically 512)
        hidden_dim: Hidden dimension for networks
        dropout: Dropout probability
    """
    
    BOND_TYPES = ['R3R3', 'R1R2', 'R1R3', 'R3R2']  # 4 classes, no 'none'
    
    def __init__(
        self,
        d_model: int = 512,
        r_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.r_dim = r_dim
        self.hidden_dim = hidden_dim
        self.num_bond_types = 4  # R3R3, R1R2, R1R3, R3R2
        
        # Context pair encoder
        self.ctx_encoder = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # R-group pair encoder (3x3 = 9 pairwise scores)
        self.r_encoder = nn.Sequential(
            nn.Linear(9, 64),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Position prediction head (outputs single score per pair)
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim + 64, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # Type prediction head (outputs 4 class logits per pair)
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim + 64, 64),
            nn.GELU(),
            nn.Linear(64, self.num_bond_types)
        )
        
    def forward(
        self,
        current_context: torch.Tensor,
        current_r: torch.Tensor,
        history_context: torch.Tensor,
        history_r: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict bonds between current token and all previous tokens.
        
        Args:
            current_context: Context for current token [B, d_model]
            current_r: R-group embeddings for current token [B, 3, r_dim]
            history_context: Context for previous tokens [B, t, d_model]
            history_r: R-group embeddings for previous tokens [B, t, 3, r_dim]
            history_mask: Mask for valid history positions [B, t]
            
        Returns:
            position_scores: [B, t] - bonding score for each history position
            type_logits: [B, t, 4] - bond type logits for each history position
        """
        batch_size = current_context.size(0)
        history_len = history_context.size(1)
        device = current_context.device
        
        if history_len == 0:
            return (
                torch.zeros(batch_size, 0, device=device),
                torch.zeros(batch_size, 0, self.num_bond_types, device=device)
            )
        
        # Expand current context for pairing with each history position
        current_ctx_exp = current_context.unsqueeze(1).expand(-1, history_len, -1)  # [B, t, d_model]
        
        # Concatenate current and history context
        ctx_pairs = torch.cat([current_ctx_exp, history_context], dim=-1)  # [B, t, d_model*2]
        
        # Encode context pairs
        ctx_feat = self.ctx_encoder(ctx_pairs)  # [B, t, hidden_dim]
        
        # Compute R-group pairwise scores
        # current_r: [B, 3, r_dim], history_r: [B, t, 3, r_dim]
        current_r_exp = current_r.unsqueeze(1).expand(-1, history_len, -1, -1)  # [B, t, 3, r_dim]
        
        # Compute 3x3 pairwise scores for each (current, history) pair
        # r_scores[b, h, i, j] = current_r[b, i] · history_r[b, h, j]
        r_scores = torch.einsum('btid,btjd->btij', current_r_exp, history_r)  # [B, t, 3, 3]
        r_scores = r_scores / (self.r_dim ** 0.5)  # Scale
        r_scores_flat = r_scores.view(batch_size, history_len, 9)  # [B, t, 9]
        
        # Encode R-group features
        r_feat = self.r_encoder(r_scores_flat)  # [B, t, 64]
        
        # Combine context and R-group features
        combined = torch.cat([ctx_feat, r_feat], dim=-1)  # [B, t, hidden_dim+64]
        
        # Predict position scores and type logits
        position_scores = self.position_head(combined).squeeze(-1)  # [B, t]
        type_logits = self.type_head(combined)  # [B, t, 4]
        
        # Apply mask if provided
        if history_mask is not None:
            position_scores = position_scores.masked_fill(~history_mask.bool(), float('-inf'))
            type_logits = type_logits.masked_fill(~history_mask.bool().unsqueeze(-1), float('-inf'))
        
        return position_scores, type_logits
    
    def predict_connection(
        self,
        current_context: torch.Tensor,
        current_r: torch.Tensor,
        history_context: torch.Tensor,
        history_r: torch.Tensor,
        threshold: float = 0.0
    ) -> Optional[Dict]:
        """
        Predict if current token connects to any previous token (inference).
        
        Args:
            current_context: [1, d_model]
            current_r: [1, 3, r_dim]
            history_context: [1, t, d_model]
            history_r: [1, t, 3, r_dim]
            threshold: Probability threshold (0-1) for predicting a bond
            
        Returns:
            Bond dict with 'res1', 'res2', 'bond_type', 'position_score', 'type_confidence'
            or None if no bond predicted
        """
        position_scores, type_logits = self.forward(
            current_context, current_r, history_context, history_r
        )
        # position_scores: [1, t], type_logits: [1, t, 4]
        
        if position_scores.size(1) == 0:
            return None
        
        # Apply sigmoid to get calibrated probabilities
        position_probs = torch.sigmoid(position_scores[0])  # [t]
        
        # Find best position
        best_pos_prob, best_pos_idx = position_probs.max(dim=0)
        best_pos_idx = best_pos_idx.item()
        best_pos_score = best_pos_prob.item()
        
        if best_pos_score < threshold:
            return None
        
        # Get bond type for best position
        best_type_logits = type_logits[0, best_pos_idx]  # [4]
        best_type_probs = F.softmax(best_type_logits, dim=-1)
        best_type_idx = best_type_logits.argmax().item()
        best_type_conf = best_type_probs[best_type_idx].item()
        
        return {
            'res1': best_pos_idx + 1,  # 1-indexed (history position)
            'bond_type': self.BOND_TYPES[best_type_idx],
            'position_score': best_pos_score,
            'type_confidence': best_type_conf
        }


class ContextOnlyAutoregressiveRingPredictor(nn.Module):
    """Autoregressive ring predictor using only the pair ``(h_t, h_j)``.

    This is the w/o R-site ablation counterpart of
    :class:`AutoregressiveRingPredictor`. It contains no R-site encoder and its
    position/type heads receive only the encoded causal-context pair.
    """

    BOND_TYPES = ['R3R3', 'R1R2', 'R1R3', 'R3R2']

    def __init__(
        self,
        d_model: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_bond_types = 4

        # MLP_H([h_t, h_j]); there is intentionally no MLP_R/r_encoder.
        self.ctx_encoder = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Distinct checkpoint names prevent accidental loading of the full
        # predictor heads, whose input dimensions include R-site features.
        self.context_position_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.context_type_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.num_bond_types),
        )

    def forward(
        self,
        current_context: torch.Tensor,
        history_context: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict ring positions and types from ``h_t`` and history ``h_j``."""
        batch_size = current_context.size(0)
        history_len = history_context.size(1)
        device = current_context.device

        if history_len == 0:
            return (
                torch.zeros(batch_size, 0, device=device),
                torch.zeros(
                    batch_size, 0, self.num_bond_types, device=device
                ),
            )

        current_expanded = current_context.unsqueeze(1).expand(
            -1, history_len, -1
        )
        context_pairs = torch.cat(
            [current_expanded, history_context], dim=-1
        )
        context_features = self.ctx_encoder(context_pairs)
        position_scores = self.context_position_head(context_features).squeeze(-1)
        type_logits = self.context_type_head(context_features)

        if history_mask is not None:
            valid_history = history_mask.bool()
            position_scores = position_scores.masked_fill(
                ~valid_history, float('-inf')
            )
            type_logits = type_logits.masked_fill(
                ~valid_history.unsqueeze(-1), float('-inf')
            )

        return position_scores, type_logits


def compute_ring_loss_ar(
    position_scores: torch.Tensor,
    type_logits: torch.Tensor,
    ring_bonds: List[List[Dict]],
    current_positions: torch.Tensor,
    margin: float = 1.0
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute ring bond loss for autoregressive prediction using BCE loss.
    
    Args:
        position_scores: [B, t] - raw logits for each history position
        type_logits: [B, t, 4] - bond type logits
        ring_bonds: List of ring bond info per sample
            Each element: [{'i': int, 'j': int, 'type': int}, ...]
            where i < j, type is 0-3 (R3R3, R1R2, R1R3, R3R2)
        current_positions: [B] - current position index for each sample
        margin: Unused (kept for API compatibility)
        
    Returns:
        total_loss: Combined position and type loss
        loss_dict: {'position_loss': ..., 'type_loss': ...}
    """
    device = position_scores.device
    batch_size = position_scores.size(0)
    history_len = position_scores.size(1)
    
    if history_len == 0:
        zero = torch.tensor(0.0, device=device)
        return zero, {'position_loss': zero, 'type_loss': zero}
    
    all_logits = []
    all_labels = []
    all_type_logits = []
    all_type_labels = []
    
    for b in range(batch_size):
        curr_pos = current_positions[b].item()
        bonds = ring_bonds[b]
        
        # Build binary label vector
        labels = torch.zeros(history_len, device=device)
        
        positive_hist_pos = -1
        bond_type = -1
        for bond in bonds:
            i, j = bond['i'], bond['j']
            if j == curr_pos and i < curr_pos:
                positive_hist_pos = i
                bond_type = bond['type']
                break
            elif i == curr_pos and j < curr_pos:
                positive_hist_pos = j
                bond_type = bond['type']
                break
        
        if positive_hist_pos >= 0:
            labels[positive_hist_pos] = 1.0
            all_type_logits.append(type_logits[b, positive_hist_pos])
            all_type_labels.append(bond_type)
        
        all_logits.append(position_scores[b])
        all_labels.append(labels)
    
    # BCE position loss with pos_weight
    cat_logits = torch.cat(all_logits)
    cat_labels = torch.cat(all_labels)
    num_pos = cat_labels.sum().clamp(min=1)
    num_neg = (cat_labels.numel() - num_pos).clamp(min=1)
    pos_weight = (num_neg / num_pos).clamp(max=50.0)
    position_loss = F.binary_cross_entropy_with_logits(
        cat_logits, cat_labels,
        pos_weight=torch.tensor(pos_weight, device=device)
    )
    
    # Type loss
    if all_type_logits:
        type_logits_t = torch.stack(all_type_logits)
        type_labels_t = torch.tensor(all_type_labels, device=device, dtype=torch.long)
        type_loss = F.cross_entropy(type_logits_t, type_labels_t)
    else:
        type_loss = torch.tensor(0.0, device=device)
    
    total_loss = position_loss + type_loss
    
    return total_loss, {
        'position_loss': position_loss,
        'type_loss': type_loss
    }
