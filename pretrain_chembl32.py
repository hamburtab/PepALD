import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import math

from helm_transformer import HELMTransformer, create_helm_transformer_for_chembl32
from helm_diffusion import HELMDiffusion, HELMSequenceDataset
from chembl32_config import CHEMBL32_CONFIG


class ChEMBL32Trainer:

    def __init__(self, config):
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
        
        print(f" 初始化完成")
        print(f" 设备: {self.device}")
        print(f" 检查点目录: {self.checkpoint_dir}")


    
    def setup_logging(self):
        """设置日志"""
        log_file = self.checkpoint_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file)]
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self):
        print(" 准备ChEMBL32数据...")
        
        if not Path(self.config.chembl32_data_file).exists():
            raise FileNotFoundError(
                f"ChEMBL32数据文件不存在: {self.config.chembl32_data_file}\n"
            )
        
        self.dataset = HELMSequenceDataset(
            data_file=self.config.chembl32_data_file,
            max_seq_len=self.config.max_seq_len,
            vocab_file=self.config.vocab_file
        )
        
        print(f"数据集加载完成:")
        print("训练序列数:", f"{len(self.dataset):,}")
        print("最大序列长度:", f"{self.config.max_seq_len}")
        print("词汇表大小:", f"{len(self.dataset.vocab)}")

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=True
        )

        print(f"数据加载器创建完成:")
        print(f"批次大小:", f"{self.config.batch_size}")
        print(f"每轮批次数:", f"{len(self.dataloader)}")

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
            use_molformer=True,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end,
            d_ff=self.config.dim_feedforward
        ).to(self.device)
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_info = {
            "总参数数": f"{total_params:,}",
            "可训练参数": f"{trainable_params:,}",
            "模型大小": f"{total_params * 4 / 1024 / 1024:.1f} MB"
        }
        print(f"   模型构建完成:")
        for key, value in model_info.items():
            print(f"   {key}: {value}")
    
    def setup_optimizer(self):
        print("   设置优化器...")
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        total_steps = len(self.dataloader) * self.config.train_epochs # 一共有多少个训练步，即多少个batch
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
        
        optimizer_info = {
            "学习率": self.config.learning_rate,
            "权重衰减": self.config.weight_decay,
            "总训练步数": total_steps
        }
        print(f"   优化器设置完成:")
        for key, value in optimizer_info.items():
            print(f"   {key}: {value}")
    
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
        # 配置参数
        val_batches = getattr(self.config, 'val_batches', 50)
        sample_count = getattr(self.config, 'sample_count', 5)
        # 默认展示全部生成的样本；如需限制数量，可在配置中设置 display_count
        display_count = getattr(self.config, 'display_count', None)
        
        # 计算验证损失
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if i >= val_batches:
                    break
                result = self.model(
                    batch['token_ids'].to(self.device),
                    mask=batch['mask'].to(self.device),
                    helm_sequences=batch['helm_sequence']
                )
                total_loss += result['loss'].item()
        
        avg_loss = total_loss / min(val_batches, len(self.dataloader))
        
        # 生成并显示样本
        self._generate_and_display_samples(sample_count, display_count)
        
        self.model.train()
        return avg_loss
    
    def _generate_and_display_samples(self, sample_count: int, display_count: int | None):
        """生成样本并显示"""
        try:
            samples = self.model.sample(
                num_samples=sample_count, 
                max_seq_len=self.config.max_seq_len, 
                predict_ring_bonds=True
            )
            
            print(f"\n 生成的 {len(samples)} 个样本序列:")
            
            if display_count is None:
                display_count = len(samples)
            else:
                display_count = min(display_count, len(samples))

            for i, sample in enumerate(samples[:display_count]): 
                if isinstance(sample, dict) and 'tokens' in sample:
                    ring_connections = sample.get('ring_connections', [])
                    helm_seq = self.dataset.decode_sequence(sample['tokens'], ring_connections)
                    ring_info = f"[环键数: {len(ring_connections)}]" if ring_connections else "[线性]"
                    print(f"   样本 {i+1} {ring_info}: {helm_seq}")
                else:
                    tokens = sample if not isinstance(sample, dict) else sample.get('tokens', sample)
                    helm_seq = self.dataset.decode_sequence(tokens)
                    print(f"   样本 {i+1}: {helm_seq}")
                
        except Exception as e:
            self.logger.warning(f"样本生成失败: {e}")
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
                x = batch['token_ids'].to(self.device)
                mask = batch['mask'].to(self.device)  # 使用mask
                helm_sequences = batch['helm_sequence']  # 这是一个列表
                
                result = self.model(x, mask=mask, helm_sequences=helm_sequences)
                loss = result['loss']
                
                self.optimizer.zero_grad()
                loss.backward()
                
                max_grad_norm = getattr(self.config, 'max_grad_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                
                self.optimizer.step()
                self.scheduler.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                
                current_lr = self.scheduler.get_last_lr()[0]
                
                # 增强日志信息，显示分别的损失
                diffusion_loss = result.get('diffusion_loss', torch.tensor(0.0))
                ring_bond_loss = result.get('ring_bond_loss', torch.tensor(0.0))
                
                pbar.set_postfix({
                    'total': f'{loss.item():.4f}',
                    'diff': f'{diffusion_loss.item():.4f}',
                    'ring': f'{ring_bond_loss.item():.4f}',
                    'avg': f'{epoch_loss/num_batches:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
                

                if self.global_step % self.config.log_interval == 0:
                    self.logger.info(
                        f"Epoch {epoch+1}, Step {self.global_step}, "
                        f"Total Loss: {loss.item():.4f}, "
                        f"Diffusion: {diffusion_loss.item():.4f}, "
                        f"Ring Bond: {ring_bond_loss.item():.4f}, "
                        f"LR: {current_lr:.2e}"
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
    """主函数：完全使用配置文件中的参数进行训练"""
    try:
        config = CHEMBL32_CONFIG
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
