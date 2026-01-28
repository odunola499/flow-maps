import os
from dataclasses import dataclass, asdict
import torch
from torch.nn import functional as F
from torch import nn, Tensor, optim
from flow_map.pytorch.data import checkerboard_loader
from flow_map.trainer import Config, BaseTrainer
from flow_map.common import plot_checkerboard
from safetensors.torch import save_file
from comet_ml import start
from datetime import datetime

@dataclass
class CheckersConfig(Config):
    n_hidden:int = 4
    n_neurons:int = 512
    output_dim:int = 2

    checkers_n_samples:int = 2000

    max_steps:int = 150000
    batch_size:int = 400
    lr:float = 1e-3
    warmup_ratio:float = 0.2
    max_valid_steps:int = 100
    valid_interval:int = 500


class FlowTrainer(BaseTrainer):
    def __init__(self,
                 model:nn.Module,
                 config:Config,
                 device = None
                 ):

        super().__init__(model, config)

        self.configure_optimizers()
        self.configure_loaders()
        self.device = device

        self.model = model.to(device)
        self.exp = start(
            project_name = 'flow-maps',
            workspace='odunola',
        )
        self.exp.log_parameters(asdict(config))

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr = self.config.lr,
            fused = True
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = self.config.max_steps
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_loaders(self):
        loader = checkerboard_loader(
            batch_size = self.config.batch_size,
            n_samples=self.config.checkers_n_samples,
            max_width=4,
            max_height=4,
        )
        self.train_loader = iter(loader)
        self.valid_loader = iter(loader)


    def compute_loss(self, data):
        x_1, width, height = data
        x_1 = x_1.to(self.device)

        batch_size = x_1.size(0)
        x_0 = torch.randn_like(x_1,device = self.device)
        t = torch.rand([batch_size], device = x_1.device)
        x_t = self.get_interpolant(x_0, x_1, t)
        flow = x_1 - x_0
        pred = self.model(x_t, t)
        loss = F.mse_loss(pred, flow)
        return loss

    def get_interpolant(self, x_0:Tensor, x_1:Tensor, t:Tensor):
        t = t.unsqueeze(1).unsqueeze(1)
        x_t = (1-t)*x_0 + t*x_1
        return x_t

    def save_ckpt(self, step):
        weights = self.model.state_dict()
        os.makedirs(self.config.ckpt_dir, exist_ok=True)
        filename = os.path.join(self.config.ckpt_dir,f'weights-{step}.safetensors')
        save_file(weights, filename)
        if self.config.save_plots:
            plots_dir = self.config.save_plots
            image_dir = os.path.join(plots_dir, f'flow-map-plots-{step}')
            os.makedirs(image_dir, exist_ok=True)
            start_path = os.path.join(image_dir, 'start.png')
            end_path = os.path.join(image_dir, 'end.png')
            results = self.model.sample(self.config.checkers_n_samples)
            plot_checkerboard(results[0].cpu().detach().numpy(), save = start_path)
            plot_checkerboard(results[-1].cpu().detach().numpy(), save = end_path)


    def train_step(self, batch):
        loss = self.compute_loss(batch)
        self.exp.log_metrics({'train_loss': loss})
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss

    def valid_step(self, batch):
        loss = self.compute_loss(batch)
        self.exp.log_metrics({'valid_loss': loss})
        return loss

if __name__ == "__main__":
    from flow_map.pytorch.models.mlp import MLP
    device = torch.device('cuda')
    model = MLP()
    config = CheckersConfig()
    trainer = FlowTrainer(model, config, device=device)

    trainer.train()