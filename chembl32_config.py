import os
from pathlib import Path


class ChEMBL32Config:
    def __init__(self):

        # 数据配置
        self.chembl32_data_file = "./data/helm_sequences_chembl32.txt"  # 实际使用的数据文件
        self.max_seq_len = 46  # 选取了chembl32中最长的序列长度
        self.vocab_file = "./data/helm_vocab.json"

        # 模型架构
        self.d_model = 512
        self.nhead = 16
        self.num_layers = 10
        self.dim_feedforward = 2048
        self.dropout = 0.15

        # diffusion参数
        self.T = 1000
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.beta_schedule = "cosine"
        
        # 训练参数
        self.train_epochs = 99
        self.batch_size = 64
        self.learning_rate = 5e-5
        self.weight_decay = 0.01
        
        # 日志
        self.checkpoint_dir = "/root/autodl-tmp/chembl32_checkpoints"
        self.log_interval = 100
        self.val_interval = 500
        self.save_every_n_steps = 1000
        
        self._create_directories()
    
    def _create_directories(self):
        directories = [
            self.checkpoint_dir, "./data/chembl32"
        ]
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save_config(self, file_path: str):
        import json
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_config(cls, file_path: str):
        import json
        config = cls()
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        return config
    
    def print_config(self):
        print("=" * 60)
        print(f" {self.project_name} 配置")
        print("=" * 60)
        
        sections = {
            "项目信息": ["project_name", "version", "description"],
            "数据配置": ["chembl32_data_file", "max_seq_len", "vocab_file"],
            "模型架构": ["d_model", "nhead", "num_layers", "dim_feedforward", "dropout"],
            "扩散模型": ["T", "beta_start", "beta_end", "beta_schedule"],
            "训练参数": ["train_epochs", "batch_size", "learning_rate", "weight_decay"],
            "输出设置": ["checkpoint_dir", "log_interval", "val_interval", "save_every_n_steps"]
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
