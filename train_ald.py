"""
Training script for Autoregressive Latent Diffusion (ALD) model.

All parameters are loaded from configs/default.json.
Modify that file to change training settings.

Usage:
    python train_ald.py
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import asdict

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
from ald.config import ALDConfig

# ============================================================
# Configuration File Path (the ONLY place to modify settings)
# ============================================================
CONFIG_FILE = PROJECT_ROOT / "configs" / "default.json"
# ============================================================


class Trainer:
    """Trainer class for ALD model."""
    
    def __init__(
        self,
        model: AutoregressiveLatentDiffusion,
        train_loader: DataLoader,
        config: ALDConfig,
        vocab: dict
    ):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.vocab = vocab
        
        train_cfg = config.training
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay
        )
        
        # Learning rate scheduler with warmup
        def lr_lambda(step):
            if step < train_cfg.warmup_steps:
                return step / train_cfg.warmup_steps
            return 1.0
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Mixed precision
        self.scaler = GradScaler() if train_cfg.use_amp else None
        self.use_amp = train_cfg.use_amp
        
        # Device
        self.device = torch.device(train_cfg.device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Checkpoint directory
        self.checkpoint_dir = Path(train_cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config to checkpoint directory
        config.save(self.checkpoint_dir / 'config.json')
        
        # Training state
        self.global_step = 0
        self.epoch = 0
    
    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        token_ids = batch['token_ids'].to(self.device)
        mask = batch['mask'].to(self.device)
        
        # Forward pass
        if self.use_amp:
            with autocast():
                loss_dict = self.model(token_ids, mask)
                loss = loss_dict['loss']
        else:
            loss_dict = self.model(token_ids, mask)
            loss = loss_dict['loss']
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.training.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )
            self.optimizer.step()
        
        self.scheduler.step()
        self.global_step += 1
        
        return loss_dict
    
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
            'config': asdict(self.config),
            'vocab': self.vocab
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"💾 Saved checkpoint to {path}")
        
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
        self.epoch = checkpoint.get('epoch', 0)
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"📂 Loaded checkpoint from {path} (step {self.global_step})")
    
    def train(self):
        """Main training loop."""
        train_cfg = self.config.training
        
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        self.config.print_summary()
        
        start_time = time.time()
        
        for epoch in range(self.epoch, train_cfg.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in self.train_loader:
                loss_dict = self.train_step(batch)
                epoch_loss += loss_dict['loss'].item()
                num_batches += 1
                
                # Logging
                if self.global_step % train_cfg.log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    
                    print(f"Step {self.global_step:6d} | "
                          f"Epoch {epoch+1:3d} | "
                          f"Loss: {loss_dict['loss'].item():.4f} | "
                          f"Avg: {avg_loss:.4f} | "
                          f"LR: {lr:.2e} | "
                          f"Time: {elapsed:.0f}s")
                
                # Save checkpoint
                if self.global_step % train_cfg.save_interval == 0 and self.global_step > 0:
                    self.save_checkpoint()
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            print(f"\n📊 Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}\n")
            
            # Save loss to log file
            log_path = self.checkpoint_dir / "training_log.txt"
            with open(log_path, "a") as f:
                f.write(f"Epoch {epoch+1}: Loss={avg_epoch_loss:.6f}, Time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Save end-of-epoch checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        
        # Save final model
        self.save_checkpoint("final_model.pt")


def main():
    """Main function."""
    
    # Load config from file
    print(f"📄 Loading config from: {CONFIG_FILE}")
    if not CONFIG_FILE.exists():
        print(f"❌ Config file not found: {CONFIG_FILE}")
        print("   Please create it or copy from configs/default.json")
        sys.exit(1)
    
    config = ALDConfig.load(str(CONFIG_FILE))
    
    # Load vocab
    with open(config.training.vocab_file, 'r') as f:
        vocab = json.load(f)
    print(f"📚 Loaded vocabulary with {len(vocab)} tokens")
    
    # Create dataset
    dataset = HELMDataset(
        data_file=config.training.train_data_file,
        vocab_file=config.training.vocab_file,
        max_seq_len=config.model.max_seq_len
    )
    print(f"📊 Loaded {len(dataset)} training sequences")
    
    # Create dataloader
    collator = HELMCollator(vocab['<PAD>'])
    train_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    # Create model
    model = AutoregressiveLatentDiffusion(vocab=vocab, config=config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Created model with {total_params/1e6:.1f}M parameters")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=config,
        vocab=vocab
    )
    
    # Check for resume
    latest_checkpoint = Path(config.training.checkpoint_dir) / 'latest.pt'
    if latest_checkpoint.exists():
        print(f"\n🔄 Found existing checkpoint, resuming training...")
        trainer.load_checkpoint(latest_checkpoint)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
