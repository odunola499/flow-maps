import os
import math
import copy
import glob
import shutil
from dataclasses import dataclass
import torch
from torch import Tensor, optim
from flow_map.pytorch.checkerboard.data import get_loader as checkerboard_loader
from flow_map.trainer import Config, BaseTrainer
from flow_map.common import plot_checkerboard
from safetensors.torch import save_file

from flow_map.pytorch.checkerboard.model import CondFlowMapMLP
from transformers import get_cosine_schedule_with_warmup


@dataclass
class CheckersConfig(Config):
    n_hidden: int = 4
    n_neurons: int = 512
    output_dim: int = 2

    checkers_n_samples: int = 5000

    max_steps: int = 50000
    batch_size: int = 16
    lr: float = 3e-4
    warmup_ratio: float = 0.2
    max_valid_steps: int = 10
    valid_interval: int = 500

    ema_decay: float = 0.999

    max_checkpoints: int = 5


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
    def __init__(self, model: CondFlowMapMLP, config: CheckersConfig, device,
                 n: float = 0.75):
        super().__init__(model, config)
        self.device = device
        self.model: CondFlowMapMLP = self.model.to(device)
        self.n = n

        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_decay = config.ema_decay

        self.configure_optimizers()
        self.config = config
        # self.exp = start(project_name='flow-maps', workspace='odunola')
        # self.exp.log_parameters(asdict(config))

    @torch.no_grad()
    def update_ema(self):
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, int(self.config.max_steps * 0.1), self.config.max_steps
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_loaders(self):
        loader = checkerboard_loader(
            batch_size=self.config.batch_size,
            n_samples=self.config.checkers_n_samples,
            max_width=5,
            max_height=5,
        )
        loader = iter(loader)
        self.train_loader = loader
        self.valid_loader = loader

    def _cleanup_old_checkpoints(self):
        weight_files = glob.glob(os.path.join(self.config.ckpt_dir, 'weights-*.safetensors'))
        ema_files = glob.glob(os.path.join(self.config.ckpt_dir, 'ema-weights-*.safetensors'))

        def extract_step(filepath):
            basename = os.path.basename(filepath)
            step_str = basename.split('-')[-1].replace('.safetensors', '')
            return int(step_str)

        weight_files.sort(key=extract_step)
        ema_files.sort(key=extract_step)

        max_ckpts = self.config.max_checkpoints
        if len(weight_files) > max_ckpts:
            for f in weight_files[:-max_ckpts]:
                os.remove(f)
        if len(ema_files) > max_ckpts:
            for f in ema_files[:-max_ckpts]:
                os.remove(f)

    def _cleanup_old_plots(self):
        if not self.config.save_plots:
            return

        plots_dir = self.config.save_plots
        plot_dirs = glob.glob(os.path.join(plots_dir, 'flow-map-plots-*'))

        def extract_step(dirpath):
            basename = os.path.basename(dirpath)
            step_str = basename.split('-')[-1]
            return int(step_str)

        plot_dirs.sort(key=extract_step)

        max_ckpts = self.config.max_checkpoints
        if len(plot_dirs) > max_ckpts:
            for d in plot_dirs[:-max_ckpts]:
                shutil.rmtree(d)

    @torch.no_grad()
    def save_ckpt(self, step):
        os.makedirs(self.config.ckpt_dir, exist_ok=True)

        weights = self.model.state_dict()
        filename = os.path.join(self.config.ckpt_dir, f'weights-{step}.safetensors')
        save_file(weights, filename)

        ema_weights = self.ema_model.state_dict()
        ema_filename = os.path.join(self.config.ckpt_dir, f'ema-weights-{step}.safetensors')
        save_file(ema_weights, ema_filename)

        self._cleanup_old_checkpoints()

        if self.config.save_plots:
            plots_dir = self.config.save_plots
            image_dir = os.path.join(plots_dir, f'flow-map-plots-{step}')
            os.makedirs(image_dir, exist_ok=True)
            start_path = os.path.join(image_dir, 'start.png')
            end_path = os.path.join(image_dir, 'end.png')
            results = self.ema_model.sample(self.config.checkers_n_samples, num_steps=2)
            x_hat = results[-1]
            x_true = next(self.train_loader)[0][0].to(self.device)
            #w1 = sliced_wasserstein(x_hat, x_true)
            plot_checkerboard(results[0].cpu().detach().numpy(), save=start_path)
            plot_checkerboard(results[-1].cpu().detach().numpy(), save=end_path)
            self._cleanup_old_plots()

    def get_interpolant(self, x_0: Tensor, x_1: Tensor, time: Tensor):
        time = time.unsqueeze(1).unsqueeze(1)
        x_t = (1 - time) * x_0 + time * x_1
        return x_t

    def compute_weight(self, s: Tensor, t: Tensor) -> Tensor:
        return torch.ones_like(s)

    def compute_time_derivative(self, I: Tensor, start_time: Tensor, end_time: Tensor, c):
        def func(t):
            dt = t - start_time
            return I + (dt[:, None, None] * self.model(I, start_time, t, c))

        X_t, D_X_t = torch.func.jvp(
            func, (end_time,), (torch.ones_like(end_time),)
        )
        return X_t, D_X_t

    def compute_lsd_residual(self, x_1, c):
        B, N, D = x_1.shape

        x_1_flat = x_1.reshape(B * N, 1, D)
        c_flat = c.unsqueeze(1).expand(-1, N).reshape(B * N)

        x_0 = torch.randn_like(x_1_flat)

        temp1 = torch.rand([B * N], device=self.device)
        temp2 = torch.rand([B * N], device=self.device)
        s = torch.minimum(temp1, temp2)
        t = torch.maximum(temp1, temp2)

        I_s = self.get_interpolant(x_0, x_1_flat, s)
        X_st, D_X_st = self.compute_time_derivative(I_s, s, t, c_flat)

        v_tt = self.ema_model(X_st.detach(), t, t, c_flat)

        error_sq = (D_X_st - v_tt).pow(2).sum(dim=-1).mean(dim=-1)
        weight = self.compute_weight(s, t)
        weighted_loss = (torch.exp(-weight) * error_sq + weight).mean()

        return weighted_loss

    def compute_flow_residual(self, x_1, c):
        B, N, D = x_1.shape

        x_1_flat = x_1.reshape(B * N, 1, D)
        c_flat = c.unsqueeze(1).expand(-1, N).reshape(B * N)

        x_0 = torch.randn_like(x_1_flat)
        t = torch.rand([B * N], device=x_1.device)

        x_t = self.get_interpolant(x_0, x_1_flat, t)
        flow = x_1_flat - x_0
        pred = self.model(x_t, t, t, c_flat)

        # Compute weighted loss for diagonal (s=t, so weight uses small epsilon)
        error_sq = (pred - flow).pow(2).sum(dim=-1).mean(dim=-1)
        weight = self.compute_weight(t, t)  # s=t for diagonal
        weighted_loss = (torch.exp(-weight) * error_sq + weight).mean()

        return weighted_loss

    def compute_loss(self, data):
        x_1, width, height = data
        x_1 = x_1.to(self.device)
        width = width.to(self.device)

        M = x_1.shape[0]
        Md = math.floor(self.n * M)
        if Md <= 0:
            Md = 1
        flow_loss = self.compute_flow_residual(x_1[:Md], width[:Md])
        lsd_loss = self.compute_lsd_residual(x_1[Md:], width[Md:])
        return flow_loss + lsd_loss, flow_loss, lsd_loss

    def train_step(self, batch, step):
        loss, flow_loss, lsd_loss = self.compute_loss(batch)
        print((loss.item(), flow_loss.item(), lsd_loss.item()))
        lr = self.optimizer.param_groups[0]['lr']
        # self.exp.log_metrics({'train_loss': loss, 'lr': lr, 'train_step': step,'train_flow_loss':flow_loss, 'train_lsd_loss':lsd_loss})
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.update_ema()

        return loss

    def valid_step(self, batch, step):
        loss, flow_loss, lsd_loss = self.compute_loss(batch)
        # self.exp.log_metrics({'valid_loss': loss, 'valid_step': step,'valid_flow_loss': flow_loss,'valid_lsd_loss':lsd_loss})
        return loss


if __name__ == "__main__":
    device = torch.device('mps')

    config = CheckersConfig()
    model = CondFlowMapMLP().to(device)
    loader = checkerboard_loader(
        batch_size=8, n_samples=2000
    )

    batch = next(iter(loader))
    trainer = LSDTrainer(model, config, device)

    #print(trainer.compute_loss(batch))
    trainer.train()
