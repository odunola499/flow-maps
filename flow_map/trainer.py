from abc import ABC, abstractmethod
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class Config:
    max_steps:int
    batch_size:int
    lr:float
    warmup_ratio:float
    max_valid_steps:int
    log_intervals:int = 10
    ckpt_dir:str = 'checkpoints'
    save_plots:str = 'plots_dir'
    valid_interval:int = 10000

class BaseTrainer(ABC):
    def __init__(self, model, config:Config, trainer_type:str = 'pytorch'):
        self.model = model
        self.config = config
        self.trainer_type = trainer_type
        self.scheduler = None
        self.optimizer = None
        self.train_loader = None
        self.valid_loader = None
        self.device = None

        self.cur_valid_step = 0

        self.configure_optimizers()
        self.configure_loaders()

    @abstractmethod
    def configure_optimizers(self):
        ...
    @abstractmethod
    def configure_loaders(self):
        ...
    @abstractmethod
    def compute_loss(self, batch):
        ...

    @abstractmethod
    def train_step(self, batch, step):
        ...
    @abstractmethod
    def valid_step(self, batch, step):
        ...

    @abstractmethod
    def save_ckpt(self, step):
        ...

    def train_loop(self):
        prog_bar = tqdm(total = self.config.max_steps)
        for step in tqdm(range(self.config.max_steps)):
            batch = next(self.train_loader)
            train_loss = self.train_step(batch, step)
            prog_bar.update(1)
            if step % self.config.log_intervals == 0:
                log_info = {
                    'mode': 'TRAIN',
                    'loss': f"{train_loss:.4f}"
                }
                prog_bar.set_postfix(log_info)
            if step % self.config.valid_interval == 0 and step != 0:
                self.valid_loop()
                self.save_ckpt(step)

    def valid_loop(self):
        self.model.eval()
        prog_bar = tqdm(total = self.config.max_valid_steps)
        for step in range(self.config.max_valid_steps):
            batch = next(self.valid_loader)
            valid_loss = self.valid_step(batch, step + self.cur_valid_step)
            prog_bar.update(1)
            if step % self.config.log_intervals == 0:
                log_info = {
                    'mode':'VALIDATION',
                    'loss':f"{valid_loss:.4f}"
                }
                prog_bar.set_postfix(log_info)

        self.cur_valid_step += self.config.max_valid_steps
        self.model.train()


    def train(self):
        print("Training...")
        if self.trainer_type == 'pytorch':
            num_params = sum([p.numel() for p in self.model.parameters()])
            print(f"Number of parameters: {num_params}")
        if self.device is not None:
            print(f"Using {self.device}")
        self.train_loop()