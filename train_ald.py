"""
Training script for Autoregressive Latent Diffusion (ALD) model.

Usage:
    python train_ald.py --config configs/ald_default.json
    python train_ald.py --data ./data/helm_sequences_chembl32.txt --epochs 100
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ald import AutoregressiveLatentDiffusion
from ald.utils.data import HELMDataset, HELMCollator
from ald.config import ALDConfig, ALDModelConfig, ALDTrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Train ALD model')
    
    # Data arguments
    parser.add_argument('--data', type=str, default='./data/helm_sequences_chembl32.txt',
                        help='Path to training data file')
    parser.add_argument('--vocab', type=str, default='./data/helm_vocab.json',
                        help='Path to vocabulary file')
    parser.add_argument('--embeddings', type=str, default='./unimol_embeddings',
                        help='Path to Uni-Mol embeddings directory')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--context_layers', type=int, default=6,
                        help='Number of context encoder layers')
    parser.add_argument('--denoiser_layers', type=int, default=4,
                        help='Number of denoiser layers')
    parser.add_argument('--diffusion_steps', type=int, default=100,
                        help='Number of diffusion steps')
    parser.add_argument('--max_seq_len', type=int, default=45,
                        help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of warmup steps')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--sample_positions', type=int, default=45,
                        help='Number of positions to sample per sequence (set to max_seq_len for full training)')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision')
    
    # Logging arguments
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log every N steps')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/ald',
                        help='Directory to save checkpoints')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


class Trainer:
    """Trainer class for ALD model."""
    
    def __init__(
        self,
        model: AutoregressiveLatentDiffusion,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        config: argparse.Namespace,
        scaler: GradScaler = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.scaler = scaler
        
        self.device = torch.device(config.device)
        self.global_step = 0
        self.epoch = 0
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.train_losses = []
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            loss = self.train_step(batch)
            epoch_loss += loss
            num_batches += 1
            
            # Logging
            if self.global_step % self.config.log_interval == 0:
                avg_loss = epoch_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Step {self.global_step} | Loss: {loss:.4f} | Avg Loss: {avg_loss:.4f} | LR: {lr:.2e}")
            
            # Save checkpoint
            if self.global_step % self.config.save_interval == 0 and self.global_step > 0:
                self.save_checkpoint()
        
        return epoch_loss / max(num_batches, 1)
    
    def train_step(self, batch):
        """Single training step."""
        self.optimizer.zero_grad()
        
        # Move data to device
        token_ids = batch['token_ids'].to(self.device)
        mask = batch['mask'].to(self.device)
        
        # Forward pass
        if self.config.amp and self.scaler is not None:
            with autocast():
                result = self.model.forward_efficient(
                    token_ids, mask,
                    sample_positions=self.config.sample_positions
                )
                loss = result['loss']
            
            # Backward pass with scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            result = self.model.forward_efficient(
                token_ids, mask,
                sample_positions=self.config.sample_positions
            )
            loss = result['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
        
        # Update scheduler
        self.scheduler.step()
        
        self.global_step += 1
        self.train_losses.append(loss.item())
        
        return loss.item()
    
    def save_checkpoint(self, filename=None):
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_step_{self.global_step}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'train_losses': self.train_losses[-1000:],  # Keep last 1000 losses
            'config': vars(self.config)
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
        # Also save as 'latest.pt'
        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.train_losses = checkpoint.get('train_losses', [])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from {path} (step {self.global_step})")
    
    def train(self, num_epochs):
        """Full training loop."""
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Total batches per epoch: {len(self.train_loader)}")
        
        start_time = time.time()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")
            
            avg_loss = self.train_epoch()
            
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch + 1} completed in {epoch_time:.1f}s | Avg Loss: {avg_loss:.4f}")
            
            # Save end-of-epoch checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        
        # Save final model
        self.save_checkpoint("final_model.pt")


def main():
    args = parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
        args.amp = False
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load vocabulary
    with open(args.vocab, 'r') as f:
        vocab = json.load(f)
    print(f"Loaded vocabulary with {len(vocab)} tokens")
    
    # Create dataset and dataloader
    dataset = HELMDataset(
        data_file=args.data,
        vocab_file=args.vocab,
        max_seq_len=args.max_seq_len
    )
    
    collator = HELMCollator(pad_id=vocab.get('<PAD>', 0))
    
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    print(f"Created dataloader with {len(train_loader)} batches")
    
    # Create model
    model = AutoregressiveLatentDiffusion(
        vocab=vocab,
        d_model=args.d_model,
        n_heads=args.n_heads,
        context_layers=args.context_layers,
        denoiser_layers=args.denoiser_layers,
        d_ff=args.d_model * 4,
        max_seq_len=args.max_seq_len,
        dropout=0.1,
        num_diffusion_steps=args.diffusion_steps,
        variance_schedule='cosine',
        embeddings_dir=args.embeddings,
        data_dir='./data'
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} total, {num_trainable:,} trainable")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Create scheduler with warmup
    total_steps = len(train_loader) * args.epochs
    
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        else:
            progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
            return max(0.1, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create scaler for AMP
    scaler = GradScaler() if args.amp else None
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=args,
        scaler=scaler
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(args.epochs)


if __name__ == '__main__':
    main()
