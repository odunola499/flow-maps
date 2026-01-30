import os
import math
from dataclasses import dataclass, asdict
import torch
from torch import nn, Tensor, optim
from flow_map.pytorch.data import checkerboard_loader
from flow_map.trainer import Config, BaseTrainer
from flow_map.common import plot_checkerboard
from safetensors.torch import save_file

from flow_map.pytorch.models.mlp import CondFlowMapMLP
from comet_ml import start

@dataclass
class CheckersConfig(Config):
    n_hidden:int = 4
    n_neurons:int = 512
    output_dim:int = 2

    checkers_n_samples:int = 2000

    max_steps:int = 50000
    batch_size:int = 10
    lr:float = 1e-3
    warmup_ratio:float = 0.2
    max_valid_steps:int = 10
    valid_interval:int = 500


@torch.no_grad()
def sliced_wasserstein(x_gen: Tensor, x_real: Tensor, n_proj: int = 100):
    device = x_gen.device
    w1 = 0.0

    for _ in range(n_proj):
        direction = torch.randn(2, device=device)
        direction = direction / direction.norm()

        proj_gen = x_gen @ direction
        proj_real = x_real @ direction

        w1 += torch.mean(
            torch.abs(
                torch.sort(proj_gen)[0] - torch.sort(proj_real)[0]
            )
        )

    return w1 / n_proj


class LSDTrainer(BaseTrainer):
    def __init__(self, model:CondFlowMapMLP, config:Config, device,
                 n:int = 0.75):
        super().__init__(model, config)
        self.device = device
        self.model:CondFlowMapMLP = self.model.to(device)
        self.n = n

        self.configure_optimizers()
        self.config = config
        #self.exp = start(project_name='flow-maps', workspace='odunola')
        #self.exp.log_parameters(asdict(config))

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            fused=True
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.max_steps
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_loaders(self):
        loader = checkerboard_loader(
            batch_size=self.config.batch_size,
            n_samples=self.config.checkers_n_samples,
            max_width=7,
            max_height=4,
        )
        loader = iter(loader)
        self.train_loader = loader
        self.valid_loader = loader

    def save_ckpt(self, step):
        weights = self.model.state_dict()
        os.makedirs(self.config.ckpt_dir, exist_ok=True)
        filename = os.path.join(self.config.ckpt_dir, f'weights-{step}.safetensors')
        save_file(weights, filename)

        if self.config.save_plots:
            plots_dir = self.config.save_plots
            image_dir = os.path.join(plots_dir, f'flow-map-plots-{step}')
            os.makedirs(image_dir, exist_ok=True)
            start_path = os.path.join(image_dir, 'start.png')
            end_path = os.path.join(image_dir, 'end.png')
            results = self.model.sample(self.config.checkers_n_samples, num_steps=10)
            x_hat = results[-1]
            x_true = next(self.train_loader)[0][0].to(self.device)
            w1 = sliced_wasserstein(x_hat, x_true)
            #self.exp.log_metrics({'w1': w1})
            plot_checkerboard(results[0].cpu().detach().numpy(), save = start_path)
            plot_checkerboard(results[-1].cpu().detach().numpy(), save = end_path)

    def get_interpolant(self, x_0:Tensor, x_1:Tensor, time:Tensor):
        time = time.unsqueeze(1).unsqueeze(1)
        x_t = (1-time)*x_0 + time*x_1
        return x_t

    def compute_time_derivative(self, I: Tensor, start_time: Tensor, end_time: Tensor, c):
        def func(t):
            dt = t - start_time
            return I + (dt[:, None, None] * self.model(I, start_time, t, c))

        X_t, D_X_t = torch.func.jvp(
            func, (end_time,), (torch.ones_like(end_time),)
        )
        return X_t, D_X_t

    def compute_lsd_residual(self, x_1, c):
        B = x_1.shape[0]

        x_0 = torch.randn_like(x_1, device=x_1.device)
        s = torch.rand([B], device=x_1.device)
        t = torch.rand([B], device=self.device)
        t = s + t * (1 - s)
        I_s = self.get_interpolant(x_0, x_1, s)
        X_st, D_X_st = self.compute_time_derivative(I_s, s, t, c)

        with torch.no_grad():
            v_tt = self.model(X_st.detach(), t,t, c)

        lsd_loss = (D_X_st - v_tt).pow(2).mean()
        #loss = torch.exp(-w_st) * lsd_loss + w_st
        return lsd_loss

    def compute_flow_residual(self, x_1, c):
        B = x_1.shape[0]
        x_0 = torch.randn_like(x_1, device=x_1.device)
        t = torch.rand([B], device=x_1.device)
        x_t = self.get_interpolant(x_0, x_1, t)
        flow = x_1 - x_0
        pred = self.model(x_t, t,t, c)
        diag_loss = (pred - flow).pow(2)
        #w_st = self.model.compute_loss_weight(torch.stack([t, t], dim=-1))[:, None, :]
        diag_loss = diag_loss.mean()
        return diag_loss

    def compute_loss(self, data):
        x_1, width, height = data
        x_1 = x_1.to(self.device)
        width = width.to(self.device)

        M = x_1.shape[0]
        Md = math.floor(self.n * M)
        if Md <=0:
            Md = 1
        flow_loss = self.compute_flow_residual(x_1[:Md], width[:Md])
        lsd_loss = self.compute_lsd_residual(x_1[Md:], width[Md:])
        return flow_loss + lsd_loss, flow_loss, lsd_loss

    def train_step(self, batch, step):
        loss, flow_loss, lsd_loss = self.compute_loss(batch)
        print((loss.item(), flow_loss.item(), lsd_loss.item()))
        lr = self.optimizer.param_groups[0]['lr']
        #self.exp.log_metrics({'train_loss': loss, 'lr': lr, 'train_step': step,'train_flow_loss':flow_loss, 'train_lsd_loss':lsd_loss})
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss

    def valid_step(self, batch, step):
        loss, flow_loss, lsd_loss = self.compute_loss(batch)
        #self.exp.log_metrics({'valid_loss': loss, 'valid_step': step,'valid_flow_loss': flow_loss,'valid_lsd_loss':lsd_loss})
        return loss



if __name__ == "__main__":
    device = torch.device('mps')

    config = CheckersConfig()
    model = CondFlowMapMLP().to(device)
    loader = checkerboard_loader(
        batch_size=8,n_samples = 2000
    )

    batch = next(iter(loader))
    trainer = LSDTrainer(model, config, device)

    #print(trainer.compute_loss(batch))
    trainer.train()
