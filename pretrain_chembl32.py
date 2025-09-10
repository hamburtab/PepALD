import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import math
import random

from helm_transformer import HELMTransformer, create_helm_transformer_for_chembl32
from helm_diffusion import HELMDiffusion, HELMSequenceDataset
from chembl32_config import CHEMBL32_CONFIG


class ChEMBL32PretrainingConfig:
    def __init__(self):
        for key, value in CHEMBL32_CONFIG.__dict__.items():
            if not key.startswith('_'):
                setattr(self, key, value)
        
        self.chembl32_data_file = "./data/helm_sequences_chembl32.txt"
        
class ChEMBL32Trainer:

    def __init__(self, config: ChEMBL32PretrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.setup_logging()
        
        self.model = None
        self.dataset = None
        self.dataloader = None
        self.optimizer = None
        self.scheduler = None
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        print(f" ChEMBL32训练器初始化完成")
        print(f" 设备: {self.device}")
        print(f" 检查点目录: {self.checkpoint_dir}")
    
    def setup_logging(self):
        """设置日志"""
        log_file = self.checkpoint_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self):
        """准备ChEMBL32数据"""
        print(" 准备ChEMBL32数据...")
        
        if not Path(self.config.chembl32_data_file).exists():
            raise FileNotFoundError(
                f"ChEMBL32数据文件不存在: {self.config.chembl32_data_file}\n"
                f"请先运行 prepare_chembl32_data.py 处理数据"
            )
        
        self.dataset = HELMSequenceDataset(
            data_file=self.config.chembl32_data_file,
            max_seq_len=self.config.max_seq_len,
            vocab_file=self.config.vocab_file
        )
        
        print(f"   数据集加载完成:")
        print(f"   训练序列数: {len(self.dataset):,}")
        print(f"   最大序列长度: {self.config.max_seq_len}")
        print(f"   词汇表大小: {len(self.dataset.vocab)}")
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=True
        )
        
        print(f"   数据加载器创建完成:")
        print(f"   批次大小: {self.config.batch_size}")
        print(f"   每轮批次数: {len(self.dataloader)}")
    
    def build_model(self):
        """构建模型"""
        print("   构建HELM扩散模型...")
        
        transformer = create_helm_transformer_for_chembl32(
            vocab_size=len(self.dataset.vocab),
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_layers=self.config.num_layers,
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.dropout
        )
        
        self.model = HELMDiffusion(
            transformer=transformer,
            vocab_size=len(self.dataset.vocab),
            T=self.config.T,
            beta_schedule=self.config.beta_schedule,
            vocab=self.dataset.vocab,  
            use_molformer=True
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"   模型构建完成:")
        print(f"   总参数数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   模型大小: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    def setup_optimizer(self):
        print("   设置优化器...")
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # 学习率调度器
        total_steps = len(self.dataloader) * self.config.train_epochs
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
        
        print(f"   优化器设置完成:")
        print(f"   学习率: {self.config.learning_rate}")
        print(f"   权重衰减: {self.config.weight_decay}")
        print(f"   总训练步数: {total_steps}")
    
    def save_checkpoint(self, epoch: int, step: int, loss: float, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config.__dict__,
            'vocab': self.dataset.vocab,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存当前检查点
        checkpoint_path = self.checkpoint_dir / f"chembl32_checkpoint_epoch_{epoch}_step_{step}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.checkpoint_dir / "chembl32_best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"保存最佳模型: {best_path}")
        
        # 保存最新模型
        latest_path = self.checkpoint_dir / "chembl32_latest_model.pth"
        torch.save(checkpoint, latest_path)
        
        self.logger.info(f"保存检查点: {checkpoint_path}")
    
    def validate_model(self) -> float:
        """验证模型并生成样本"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if i >= 50:  # 只使用50个批次进行验证
                    break
                    
                x = batch.to(self.device)
                loss = self.model.compute_loss(x)
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        # 生成样本序列
        try:
            samples = self.model.sample(num_samples=5, max_seq_len=self.config.max_seq_len)
            
            print("\n 生成的样本序列:")
            for i, sample in enumerate(samples[:3]): 
                helm_seq = self.dataset.decode_sequence(sample)
                print(f"   样本 {i+1}: {helm_seq[:100]}...")
                
        except Exception as e:
            self.logger.warning(f"样本生成失败: {e}")
        
        self.model.train()
        return avg_loss
    
    def train(self):
        """主训练循环"""
        print("   开始ChEMBL32预训练...")
        
        config_path = self.checkpoint_dir / "chembl32_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        self.model.train()
        
        for epoch in range(self.config.train_epochs):
            self.current_epoch = epoch
            epoch_loss = 0
            num_batches = 0
            
            pbar = tqdm(
                self.dataloader, 
                desc=f"Epoch {epoch+1}/{self.config.train_epochs}",
                leave=False
            )
            
            for batch_idx, batch in enumerate(pbar):
                x = batch.to(self.device)
                loss = self.model.compute_loss(x)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                
                current_lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{epoch_loss/num_batches:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
                

                if self.global_step % self.config.log_interval == 0:
                    self.logger.info(
                        f"Epoch {epoch+1}, Step {self.global_step}, "
                        f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}"
                    )
                
                if self.global_step % self.config.val_interval == 0:
                    val_loss = self.validate_model()
                    self.logger.info(f"验证损失: {val_loss:.4f}")
                
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_checkpoint(epoch, self.global_step, val_loss, is_best=True)
                

                if self.global_step % self.config.save_every_n_steps == 0:
                    self.save_checkpoint(epoch, self.global_step, loss.item())
            

            avg_epoch_loss = epoch_loss / num_batches
            print(f"\n  Epoch {epoch+1} 完成:")
            print(f"   平均损失: {avg_epoch_loss:.4f}")
            print(f"   全局步数: {self.global_step}")
            
            self.save_checkpoint(epoch, self.global_step, avg_epoch_loss)
    
        print("  ChEMBL32预训练完成!")
        print(f"  检查点保存在: {self.checkpoint_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ChEMBL32 HELM扩散模型预训练")
    parser.add_argument("--data_file", 
                       default="./data/helm_sequences_chembl32.txt",
                       help="ChEMBL32 HELM序列文件")
    parser.add_argument("--epochs", type=int, default=10,
                       help="训练轮次")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="学习率")
    parser.add_argument("--max_seq_len", type=int, default=150,
                       help="最大序列长度")
    
    args = parser.parse_args()
    
    try:
        config = ChEMBL32PretrainingConfig()
        config.chembl32_data_file = args.data_file
        config.train_epochs = args.epochs
        config.batch_size = args.batch_size
        config.learning_rate = args.learning_rate
        config.max_seq_len = args.max_seq_len
        
        trainer = ChEMBL32Trainer(config)
        
        trainer.prepare_data()
        

        trainer.build_model()
        

        trainer.setup_optimizer()
        
        trainer.train()
        
    except Exception as e:
        print(f"  训练失败: {e}")
        logging.exception("训练过程中发生错误")


if __name__ == "__main__":
    main()
