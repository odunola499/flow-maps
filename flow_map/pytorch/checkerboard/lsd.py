import os
import copy
import glob
import shutil
from dataclasses import dataclass, asdict
import torch
from torch import Tensor, optim
from flow_map.pytorch.checkerboard.data import get_loader as checkerboard_loader
from flow_map.trainer import Config, BaseTrainer
from flow_map.common import plot_checkerboard
from safetensors.torch import save_file

from flow_map.pytorch.checkerboard.model import CondFlowMapMLP
from transformers import get_cosine_schedule_with_warmup
import wandb


@dataclass
class CheckersConfig(Config):
    n_hidden: int = 4
    n_neurons: int = 512
    output_dim: int = 2

    checkers_n_samples: int = 1000

    max_steps: int = 50000
    batch_size: int = 16
    lr: float = 1e-3
    warmup_ratio: float = 0.2
    max_valid_steps: int = 10
    valid_interval: int = 1000

    ema_decay: float = 0

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
        self.Md = int(n * self.config.batch_size)

        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_decay = config.ema_decay

        self.configure_optimizers()
        self.config = config
        wandb.init(project='flow maps', entity='jenrola2292', config =
                   asdict(config))

        if self.device.type == 'cuda':
            print('Using torch compile')
            self.compute_loss = torch.compile(self._compute_loss)
        else:
            self.compute_loss = self._compute_loss

    @torch.no_grad()
    def update_ema(self):
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def configure_optimizers(self):
        optimizer = optim.RAdam(
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
            results = self.ema_model.sample(self.config.checkers_n_samples, num_steps=50)
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

        x_0 = torch.randn_like(x_1)

        temp1 = torch.rand([B], device=self.device)
        temp2 = torch.rand([B], device=self.device)
        s = torch.minimum(temp1, temp2)
        t = torch.maximum(temp1, temp2)

        I_s = self.get_interpolant(x_0, x_1, s)
        X_st, D_X_st = self.compute_time_derivative(I_s, s, t, c)

        with torch.no_grad():
            v_tt = self.ema_model(X_st, t, t, c)

        error_sq = (D_X_st - v_tt).pow(2).mean()

        weight = self.compute_weight(s, t)
        weighted_loss = (torch.exp(-weight) * error_sq + weight).mean()

        return error_sq

    def compute_flow_residual(self, x_1, c):
        B, N, D = x_1.shape

        x_0 = torch.randn_like(x_1)
        t = torch.rand([B], device=x_1.device)

        x_t = self.get_interpolant(x_0, x_1, t)
        flow = x_1 - x_0
        pred = self.model(x_t, t, t, c)
        error_sq = (pred - flow).pow(2).sum(dim=-1).mean(dim=-1)
        error = torch.nn.functional.mse_loss(pred, flow)

        weight = self.compute_weight(t, t)
        weighted_loss = (torch.exp(-weight) * error_sq + weight).mean()

        return error.mean()

    def _compute_loss(self, data):
        x_1, width, height = data
        Md = self.Md

        flow_loss = self.compute_flow_residual(x_1[:Md], width[:Md])
        #lsd_loss = self.compute_lsd_residual(x_1[Md:], width[Md:])
        lsd_loss = torch.zeros_like(flow_loss)
        return flow_loss + lsd_loss, flow_loss, lsd_loss

    def train_step(self, batch, step):
        batch = tuple(b.to(self.device) for b in batch)
        loss, flow_loss, lsd_loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()
        self.update_ema()

        lr = self.optimizer.param_groups[0]['lr']
        wandb.log({
            'train/loss': loss.item(),
            'global/lr': lr,
            'train/step': step,
            'train/flow_loss': flow_loss.item(),
            'train/lsd_loss': lsd_loss.item()
        })
        return loss

    def valid_step(self, batch, step):
        batch = tuple(b.to(self.device) for b in batch)
        loss, flow_loss, lsd_loss = self.compute_loss(batch)
        wandb.log({
            'valid/loss': loss.item(),
            'valid/step': step,
            'valid/flow_loss': flow_loss.item(),
            'valid/lsd_loss': lsd_loss.item()
        })
        return loss


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')
    config = CheckersConfig()
    model = CondFlowMapMLP().to(device)

    loader = checkerboard_loader(
        batch_size=2048, n_samples=2000
    )

    batch = next(iter(loader))
    trainer = LSDTrainer(model, config, device)

    # print(trainer.compute_loss(batch))
    trainer.train()