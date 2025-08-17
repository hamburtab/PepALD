"""
ChEMBL32项目配置文件
"""

import os
from pathlib import Path


class ChEMBL32Config:
    """ChEMBL32专用配置类"""
    
    def __init__(self):
        # 项目基本信息
        self.project_name = "HELM_Diffusion_ChEMBL32"
        self.version = "2.0.0"
        self.description = "基于ChEMBL32生物治疗药物数据的HELM扩散模型"
        
        # 数据路径
        self.chembl32_raw_file = "./data/chembl32/biotherapeutics_dict_prot_flt.csv"
        self.chembl32_processed_file = "./data/helm_sequences_chembl32.txt"
        self.chembl32_stats_file = "./data/chembl32_processing_stats.json"
        
        # 数据处理参数
        self.min_seq_len = 3        
        self.max_seq_len = 150      
        self.vocab_file = "./data/helm_vocab.json"
        self.train_ratio = 0.9      
        self.val_ratio = 0.1        

        # Transformer配置
        self.d_model = 512
        self.nhead = 8
        self.num_layers = 10
        self.dim_feedforward = 2048
        self.dropout = 0.15
        self.activation = "gelu"
        
        # 扩散模型配置
        self.T = 1000
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.beta_schedule = "cosine"
        self.loss_type = "mse"
        self.predict_type = "noise"
        
        # 训练配置
        self.train_epochs = 15
        self.batch_size = 24
        self.learning_rate = 3e-5
        self.weight_decay = 0.01
        self.grad_clip_norm = 1.0
        self.optimizer_type = "adamw"
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.eps = 1e-8
        self.scheduler_type = "cosine"
        self.warmup_steps = 2000
        self.min_lr = 1e-6
        
        # 训练策略
        self.use_data_augmentation = True
        self.augment_prob = 0.1
        self.use_mixed_precision = True
        self.grad_scale_init = 65536
        self.use_ema = True
        self.ema_decay = 0.999
        
        # 输出目录
        self.output_dir = "./chembl32_outputs"
        self.checkpoint_dir = "./chembl32_checkpoints"
        self.log_dir = "./chembl32_logs"
        self.sample_dir = "./chembl32_samples"
        
        # 保存和日志
        self.save_every_n_epochs = 1
        self.save_every_n_steps = 2000
        self.keep_last_n_checkpoints = 5
        self.log_interval = 100
        self.val_interval = 1000
        self.sample_interval = 2000
        self.tensorboard_log = True
        
        # 验证和采样
        self.val_batch_size = 32
        self.val_num_batches = 50
        self.sample_num_sequences = 10
        self.sample_algorithm = "ddim"
        self.ddim_steps = 50
        self.ddim_eta = 0.0
        
        # 计算资源
        self.device = "auto"
        self.num_workers = 8
        self.pin_memory = True
        self.gradient_checkpointing = True
        self.dataloader_drop_last = True
        
        # 实验设置
        self.experiment_name = "chembl32_v1"
        self.seed = 42
        self.deterministic = True
        self.debug_mode = False
        self.profile_training = False
        self.fast_dev_run = False
        
        # ChEMBL32特定功能
        self.analyze_chembl32_properties = True
        self.validate_generated_molecules = True
        self.compute_molecular_metrics = True
        self.post_process_sequences = True
        self.filter_invalid_sequences = True
        
        # 确保目录存在
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.output_dir, self.checkpoint_dir, self.log_dir, 
            self.sample_dir, "./data/chembl32"
        ]
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self):
        """转换为字典格式"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save_config(self, file_path: str):
        """保存配置到JSON文件"""
        import json
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_config(cls, file_path: str):
        """从JSON文件加载配置"""
        import json
        config = cls()
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        return config
    
    def print_config(self):
        """打印配置信息"""
        print("=" * 60)
        print(f"🔧 {self.project_name} 配置")
        print("=" * 60)
        
        sections = {
            "项目信息": ["project_name", "version", "description"],
            "数据配置": ["chembl32_raw_file", "max_seq_len", "min_seq_len", "train_ratio"],
            "模型架构": ["d_model", "nhead", "num_layers", "T", "beta_schedule"],
            "训练参数": ["train_epochs", "batch_size", "learning_rate", "scheduler_type"],
            "输出设置": ["checkpoint_dir", "log_interval", "save_every_n_epochs"],
            "采样配置": ["sample_num_sequences", "sample_algorithm", "ddim_steps"]
        }
        
        for section_name, keys in sections.items():
            print(f"\n📋 {section_name}:")
            for key in keys:
                if hasattr(self, key):
                    value = getattr(self, key)
                    print(f"   {key}: {value}")
        
        print("=" * 60)


# 创建全局配置实例
CHEMBL32_CONFIG = ChEMBL32Config()
