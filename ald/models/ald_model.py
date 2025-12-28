"""
Autoregressive Latent Diffusion (ALD) Model.

The main model that combines:
    1. Causal Context Encoder - processes history
    2. Diffusion Engine - generates each token via diffusion
    3. Token Mapper - maps embeddings to discrete monomers
    4. Ring Bond Predictor - predicts cyclic connections

This is the complete system for token-by-token peptide generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Literal
from pathlib import Path

from .context_encoder import CausalContextEncoder
from .token_mapper import TokenMapper
from .ring_predictor import RingBondPredictor, AutoregressiveRingPredictor
from ..diffusion.engine import DiffusionEngine
from ..core.embeddings import UniMolEmbeddingLoader


class AutoregressiveLatentDiffusion(nn.Module):
    """
    Autoregressive Latent Diffusion Model for HELM Peptide Generation.
    
    Architecture Overview:
        
        For each position t in the sequence:
            1. Context Encoding: h_t = ContextEncoder([x_0, ..., x_{t-1}])
            2. Diffusion Sampling: z_t = DiffusionEngine.sample(context=h_t)
            3. Token Mapping: x_t = TokenMapper(z_t, position=t)
            4. Ring Prediction: bonds_t = RingPredictor(h_t, x_t)
    
    The diffusion process at each step generates a single token embedding
    by denoising from Gaussian noise, conditioned on the context.
    
    Args:
        vocab: Dictionary mapping monomer symbols to indices
        embedding_dim: Dimension of Uni-Mol embeddings
        d_model: Model dimension for transformer
        n_heads: Number of attention heads
        context_layers: Number of layers in context encoder
        denoiser_layers: Number of layers in diffusion denoiser
        d_ff: Feed-forward dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        num_diffusion_steps: Number of diffusion steps (K)
        variance_schedule: Type of variance schedule
        beta_start: Starting beta value
        beta_end: Ending beta value
        embeddings_dir: Directory for Uni-Mol embeddings
        data_dir: Directory for vocab and data files
    """
    
    def __init__(
        self,
        vocab: Dict[str, int],
        embedding_dim: int = 512,
        d_model: int = 512,
        n_heads: int = 8,
        context_layers: int = 6,
        denoiser_layers: int = 4,
        d_ff: int = 2048,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        num_diffusion_steps: int = 100,
        variance_schedule: Literal['linear', 'cosine'] = 'cosine',
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        embeddings_dir: str = "./unimol_embeddings",
        data_dir: str = "./data"
    ):
        super().__init__()
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_diffusion_steps = num_diffusion_steps
        
        # Reverse vocab for decoding
        self.idx_to_token = {v: k for k, v in vocab.items()}
        
        # PAD token ID
        self.pad_id = vocab.get('<PAD>', self.vocab_size - 1)
        
        # 1. Context Encoder (The "Brain")
        self.context_encoder = CausalContextEncoder(
            embedding_dim=embedding_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=context_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            embeddings_dir=embeddings_dir,
            freeze_embeddings=True
        )
        
        # Update embedding_dim from actual loaded embeddings
        actual_embed_dim = self.context_encoder.embedding.embedding_dim
        if actual_embed_dim != embedding_dim:
            self.embedding_dim = actual_embed_dim
            print(f"[ALD] Updated embedding_dim to {actual_embed_dim}")
        
        # 2. Diffusion Engine (The "Brush")
        self.diffusion_engine = DiffusionEngine(
            embedding_dim=self.embedding_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=denoiser_layers,
            d_ff=d_ff,
            dropout=dropout,
            num_diffusion_steps=num_diffusion_steps,
            variance_schedule=variance_schedule,
            beta_start=beta_start,
            beta_end=beta_end
        )
        
        # 3. Token Mapper
        self.token_mapper = TokenMapper(
            vocab=vocab,
            embeddings_dir=embeddings_dir,
            data_dir=data_dir,
            use_embedding_norm=True,
            use_freq_weight=True,
            use_temperature_sampling=False
        )
        
        # 4. Ring Bond Predictor
        self.ring_predictor = RingBondPredictor(
            d_model=d_model,
            hidden_dim=d_model // 2,
            num_bond_types=5
        )
        
        # 5. Autoregressive Ring Predictor (for step-by-step prediction)
        self.ar_ring_predictor = AutoregressiveRingPredictor(
            d_model=d_model,
            hidden_dim=d_model // 2,
            num_bond_types=5
        )
        
        print(f"[ALD] Model initialized:")
        print(f"  - Embedding dim: {self.embedding_dim}")
        print(f"  - Model dim: {d_model}")
        print(f"  - Context layers: {context_layers}")
        print(f"  - Denoiser layers: {denoiser_layers}")
        print(f"  - Diffusion steps: {num_diffusion_steps}")
        print(f"  - Variance schedule: {variance_schedule}")
        
    def forward(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.
        
        For training, we have ground truth tokens. We:
            1. Get embeddings for all tokens
            2. For each position t, compute context from [0, t-1]
            3. Train diffusion to denoise the embedding at position t
        
        Args:
            token_ids: Ground truth token IDs [batch_size, seq_len]
            mask: Sequence mask [batch_size, seq_len]
            
        Returns:
            Dictionary with 'loss', 'diffusion_loss', 'ring_bond_loss'
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        
        # Get ground truth embeddings
        gt_embeddings = self.context_encoder.get_token_embedding(token_ids)
        # [batch_size, seq_len, embedding_dim]
        
        total_diffusion_loss = 0.0
        num_positions = 0
        
        # For each position, train diffusion conditioned on context
        for t in range(seq_len):
            # Skip PAD positions
            if mask is not None:
                valid_mask = mask[:, t] > 0
                if not valid_mask.any():
                    continue
            else:
                valid_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            
            # Get context from previous positions
            if t == 0:
                # No history - use start token context
                context = self.context_encoder.get_context_for_next_token(
                    gt_embeddings[:, :0, :]  # Empty history
                )
            else:
                # History: [0, t-1]
                history_emb = gt_embeddings[:, :t, :]
                history_mask = mask[:, :t] if mask is not None else None
                context = self.context_encoder.get_context_for_next_token(
                    history_emb, history_mask
                )
            
            # Context: [batch_size, d_model] -> [batch_size, 1, d_model]
            context = context.unsqueeze(1)
            
            # Target embedding at position t
            target_emb = gt_embeddings[valid_mask, t, :]  # [valid_batch, embedding_dim]
            context_valid = context[valid_mask]  # [valid_batch, 1, d_model]
            
            # Train diffusion
            diff_result = self.diffusion_engine.training_step(
                target_emb, context_valid
            )
            
            total_diffusion_loss += diff_result['loss'] * valid_mask.sum()
            num_positions += valid_mask.sum().item()
        
        # Average diffusion loss
        diffusion_loss = total_diffusion_loss / max(num_positions, 1)
        
        # Ring bond prediction loss
        # Get full context vectors
        contexts = self.context_encoder(token_ids, mask)  # [batch_size, seq_len, d_model]
        
        # TODO: Add ring bond supervision from HELM sequences
        ring_bond_loss = torch.tensor(0.0, device=device)
        
        return {
            'loss': diffusion_loss + 0.1 * ring_bond_loss,
            'diffusion_loss': diffusion_loss,
            'ring_bond_loss': ring_bond_loss
        }
    
    def forward_efficient(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sample_positions: int = 5
    ) -> Dict[str, torch.Tensor]:
        """
        Efficient training by sampling random positions instead of all.
        
        Args:
            token_ids: [batch_size, seq_len]
            mask: [batch_size, seq_len]
            sample_positions: Number of positions to sample per sequence
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        
        # Get ground truth embeddings
        gt_embeddings = self.context_encoder.get_token_embedding(token_ids)
        
        # Get full contexts (computed once)
        full_contexts = self.context_encoder(token_ids, mask)
        
        # Sample random positions for each batch
        if mask is not None:
            lengths = mask.sum(dim=1).long()
        else:
            lengths = torch.full((batch_size,), seq_len, device=device)
        
        total_loss = 0.0
        count = 0
        
        for b in range(batch_size):
            seq_length = lengths[b].item()
            if seq_length <= 1:
                continue
            
            # Sample positions
            num_samples = min(sample_positions, seq_length)
            positions = torch.randperm(seq_length, device=device)[:num_samples]
            
            for pos in positions:
                pos = pos.item()
                
                # Get context for this position (use previous position's output)
                if pos == 0:
                    # For first position, use zero context or learnable start
                    context = torch.zeros(1, 1, self.d_model, device=device)
                else:
                    context = full_contexts[b:b+1, pos-1:pos, :]
                
                # Target embedding
                target = gt_embeddings[b:b+1, pos, :]
                
                # Diffusion loss
                diff_result = self.diffusion_engine.training_step(target, context)
                total_loss += diff_result['loss']
                count += 1
        
        avg_loss = total_loss / max(count, 1)
        
        return {
            'loss': avg_loss,
            'diffusion_loss': avg_loss,
            'ring_bond_loss': torch.tensor(0.0, device=device)
        }
    
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        max_length: int,
        min_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        use_ddim: bool = False,
        ddim_steps: int = 50,
        predict_ring_bonds: bool = True,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Generate peptide sequences autoregressively.
        
        Args:
            num_samples: Number of sequences to generate
            max_length: Maximum sequence length
            min_length: Minimum sequence length (for random length)
            device: Device for generation
            use_ddim: Use DDIM sampling (faster)
            ddim_steps: Number of DDIM steps if use_ddim=True
            predict_ring_bonds: Whether to predict ring bonds
            verbose: Print progress
            
        Returns:
            List of dictionaries with 'tokens', 'embeddings', 'ring_connections'
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        
        # Determine target lengths
        if min_length is not None:
            lengths = torch.randint(min_length, max_length + 1, (num_samples,))
        else:
            lengths = torch.full((num_samples,), max_length)
        
        results = []
        
        for sample_idx in range(num_samples):
            target_length = lengths[sample_idx].item()
            
            # Storage for generated sequence
            generated_embeddings = []
            generated_tokens = []
            ring_connections = []
            
            if verbose:
                print(f"Generating sample {sample_idx + 1}/{num_samples}, length={target_length}")
            
            for t in range(target_length):
                # 1. Get context from history
                if t == 0:
                    # Empty history
                    history = torch.zeros(1, 0, self.embedding_dim, device=device)
                    context = self.context_encoder.get_context_for_next_token(history)
                else:
                    # Stack embeddings: each is [embedding_dim], stack to [t, embedding_dim], unsqueeze to [1, t, embedding_dim]
                    history = torch.stack(generated_embeddings, dim=0).unsqueeze(0)  # [1, t, embedding_dim]
                    context = self.context_encoder.get_context_for_next_token(history)
                
                # Context: [1, d_model] -> [1, 1, d_model]
                context = context.unsqueeze(1)
                
                # 2. Generate token embedding via diffusion
                if use_ddim:
                    embedding = self.diffusion_engine.sample_ddim(
                        batch_size=1,
                        context=context,
                        device=device,
                        num_inference_steps=ddim_steps
                    )
                else:
                    embedding = self.diffusion_engine.sample(
                        batch_size=1,
                        context=context,
                        device=device
                    )
                
                # 3. Map to token
                token_id = self.token_mapper(
                    embedding, position=t, seq_len=target_length
                )
                
                generated_embeddings.append(embedding.squeeze(0).squeeze(0))  # Remove batch and seq dims -> [embedding_dim]
                generated_tokens.append(token_id.item())
                
                # 4. Predict ring bonds (if at least 2 tokens generated)
                if predict_ring_bonds and len(generated_embeddings) >= 2:
                    # Stack history embeddings (all but current): each is [embedding_dim]
                    # Stack to [t-1, embedding_dim], unsqueeze to [1, t-1, embedding_dim]
                    history_stack = torch.stack(generated_embeddings[:-1], dim=0).unsqueeze(0)
                    history_contexts = self.context_encoder.forward_with_embeddings(history_stack)
                    
                    bond = self.ar_ring_predictor.predict_connection(
                        context.squeeze(1),  # Current context [1, d_model]
                        history_contexts,     # [1, history_len, d_model]
                        threshold=0.5
                    )
                    
                    if bond is not None:
                        bond['res2'] = t + 1  # Current position (1-indexed)
                        ring_connections.append(bond)
                
                if verbose and (t + 1) % 10 == 0:
                    print(f"  Generated {t + 1}/{target_length} tokens")
            
            # Compile result
            results.append({
                'tokens': torch.tensor(generated_tokens, device=device),
                'embeddings': torch.stack(generated_embeddings, dim=0),
                'ring_connections': ring_connections,
                'length': target_length
            })
        
        return results
    
    def decode_to_helm(
        self,
        tokens: torch.Tensor,
        ring_connections: Optional[List[Dict]] = None
    ) -> str:
        """
        Decode token IDs to HELM string.
        
        Args:
            tokens: Token IDs [seq_len]
            ring_connections: List of ring bond dictionaries
            
        Returns:
            HELM string
        """
        # Convert tokens to monomer symbols
        symbols = []
        for token_id in tokens.tolist():
            if token_id == self.pad_id:
                break
            symbol = self.idx_to_token.get(token_id, '?')
            symbols.append(symbol)
        
        if not symbols:
            return "PEPTIDE1{}$$$$"
        
        # Build sequence part
        sequence_part = f"PEPTIDE1{{{'.'.join(symbols)}}}"
        
        # Build connection part
        if ring_connections:
            conn_strings = []
            for conn in ring_connections:
                res1 = conn['res1']
                res2 = conn['res2']
                bond_type = conn['bond_type']
                
                if bond_type == 'R3R3':
                    conn_str = f"PEPTIDE1,PEPTIDE1,{res1}:R3-{res2}:R3"
                elif bond_type == 'R1R2':
                    conn_str = f"PEPTIDE1,PEPTIDE1,{res1}:R1-{res2}:R2"
                elif bond_type == 'R1R3':
                    conn_str = f"PEPTIDE1,PEPTIDE1,{res1}:R1-{res2}:R3"
                elif bond_type == 'R3R2':
                    conn_str = f"PEPTIDE1,PEPTIDE1,{res1}:R3-{res2}:R2"
                else:
                    continue
                conn_strings.append(conn_str)
            
            if conn_strings:
                return f"{sequence_part}${'|'.join(conn_strings)}$$$"
        
        return f"{sequence_part}$$$$"
    
    def generate_helm_sequences(
        self,
        num_samples: int,
        max_length: int,
        **kwargs
    ) -> List[str]:
        """
        Convenience method to generate HELM strings directly.
        
        Args:
            num_samples: Number of sequences
            max_length: Maximum length
            **kwargs: Additional arguments for sample()
            
        Returns:
            List of HELM strings
        """
        results = self.sample(num_samples, max_length, **kwargs)
        
        helm_sequences = []
        for result in results:
            helm = self.decode_to_helm(
                result['tokens'],
                result.get('ring_connections', [])
            )
            helm_sequences.append(helm)
        
        return helm_sequences
