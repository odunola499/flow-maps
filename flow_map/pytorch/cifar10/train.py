import os
from dataclasses import dataclass, asdict
from typing import Optional
import torch
from torch import nn
from torch import optim
from flow_map.pytorch.cifar10.unet import Unet
from flow_map.pytorch.cifar10.data import get_loaders, denormalize
from flow_map.trainer import BaseTrainer, Config
from flow_map.pytorch.checkerboard.flow_base import plot_checkerboard, sliced_wasserstein
from transformers import get_cosine_schedule_with_warmup
import wandb
import copy
from safetensors.torch import save_file
from tqdm.auto import tqdm
from torchvision.utils import save_image
import math

PI = math.pi

@dataclass
class CifarConfig(Config):
    num_layers:int = 6

    max_steps: int = 300000
    batch_size: int = 256
    lr: float = 5e-4
    warmup_ratio: float = 0.2
    max_valid_steps: int = 10
    valid_interval: int = 20000
    loss_ema_beta: float = 0.98
    spike_factor: float = 3.0
    spike_warmup: int = 50
    spike_abs_threshold: Optional[float] = None
    spike_dir: str = 'spike_samples'

    ema_decay: float = 0.999

    max_checkpoints: int = 5

    use_wandb: bool = False
    iterp_type:str = 'polynomial' # can be linear or trig or polynomial

    num_workers: int = 4
    use_compile: bool = True


class FlowTrainer(BaseTrainer):
    def __init__(
            self,
            model:Unet,
            config:CifarConfig,
            device
    ):
        super().__init__(
            model = model,
            config = config
        )
        self.device = device
        self.ema_model = copy.deepcopy(self.model).eval()
        self.ema_decay = config.ema_decay
        self.use_wandb = config.use_wandb
        self.loss_ema = None
        self.config = config
        self.beta_sampler = torch.distributions.Beta(2.0, 2.0)
        self._amp_enabled = (device.type == 'cuda')
        self.scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)

        if config.use_compile and device.type == 'cuda':
            self.model = torch.compile(self.model)

        if self.use_wandb:
            wandb.init(
                project='flow maps',
                entity='jenrola2292',
                name=f'cifar10-{config.iterp_type}',
                config=asdict(config)
            )

    def log(self, metrics: dict):
        if self.use_wandb:
            wandb.log(metrics)
        else:
            print(" | ".join(f"{k}: {v}" for k, v in metrics.items()))

    @torch.no_grad()
    def update_ema(self):
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def configure_loaders(self):
        train, valid = get_loaders(self.config.batch_size, num_workers=self.config.num_workers)
        self.train_loader = iter(train)
        self.valid_loader = iter(valid)
        self.train_dataset = train
        self.valid_dataset = valid

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

    def compute_interpolant(self, x0, x1,t):
        t = t[:,None, None,None]
        if self.config.iterp_type == 'linear':
            return ((1-t)*x0)+(t*x1)
        elif self.config.iterp_type == 'trig':
            a = (1+torch.cos(PI * t)) / 2
            b = (1-torch.cos(PI * t)) / 2
            return (a*x0) + (b*x1)
        elif self.config.iterp_type == 'polynomial':
            b = (3*(t)**2) - (2*(t**3))
            a = 1-b
            return (a * x0) + (b * x1)
        else:
            raise NotImplementedError


    def compute_target(self, x0, x1, t):
        if self.config.iterp_type == 'linear':
            return x1 - x0
        elif self.config.iterp_type == 'trig':
            t = t[:, None, None, None]
            return ((PI * torch.sin(PI * t)) / 2) * (x1-x0)
        elif self.config.iterp_type == 'polynomial':
            t = t[:, None, None, None]
            return 6*(t -t**2)*(x1 - x0)
        else:
            raise NotImplementedError

    def sample_time(self, B, device, dtype):
        if self.config.iterp_type == 'linear':
            return torch.rand([B], device=device, dtype=dtype)
        else:
            t = self.beta_sampler.sample([B]).to(device=device, dtype=dtype)
            return t.clamp(1e-5, 1 - 1e-5)


    def compute_loss(self, batch):
        x1, label = batch
        B = x1.shape[0]

        x0 = torch.randn_like(x1)
        t = self.sample_time(B,x1.device, x1.dtype)
        xt = self.compute_interpolant(x0, x1, t)
        target = self.compute_target(x0, x1, t)

        pred = self.model(
            x = xt,
            timestep = t,
            y = label
        )

        loss = nn.functional.mse_loss(
            pred, target
        )
        return loss

    def _update_loss_ema(self, loss_value: float):
        if self.loss_ema is None:
            self.loss_ema = loss_value
        else:
            beta = self.config.loss_ema_beta
            self.loss_ema = self.loss_ema * beta + loss_value * (1.0 - beta)
        return self.loss_ema

    def _maybe_save_spike(self, batch, step: int, loss_value: float, prev_ema: Optional[float]):
        if prev_ema is None:
            return
        if step < self.config.spike_warmup:
            return
        if self.config.spike_abs_threshold is not None and loss_value < self.config.spike_abs_threshold:
            return
        if loss_value < prev_ema * self.config.spike_factor:
            return

        os.makedirs(self.config.spike_dir, exist_ok=True)
        x1, label = batch
        x1_cpu = x1[:4].detach().float().cpu()
        label_cpu = label[:4].detach().cpu().tolist()
        filename = os.path.join(
            self.config.spike_dir,
            f"spike-step-{step}-loss-{loss_value:.4f}-ema-{prev_ema:.4f}-labels-{label_cpu}.png",
        )
        save_image(denormalize(x1_cpu), filename)

    def train_step(self, batch, step):
        batch = tuple(b.to(self.device, non_blocking=True) for b in batch)

        with torch.autocast(device_type=self.device.type,enabled=self._amp_enabled):
            loss = self.compute_loss(batch)

        loss_value = loss.item()
        prev_ema = self.loss_ema
        self._update_loss_ema(loss_value)
        self._maybe_save_spike(batch, step, loss_value, prev_ema)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.update_ema()

        lr = self.optimizer.param_groups[0]['lr']
        self.log({
            'train/loss': loss_value,
            'train/loss_ema': self.loss_ema,
            'global/lr': lr,
            'train/step': step
        })
        return loss

    def valid_step(self, batch, step):
        batch = tuple(b.to(self.device, non_blocking=True) for b in batch)
        with torch.autocast(device_type=self.device.type,enabled=self._amp_enabled):
            loss = self.compute_loss(batch)

        self.log({
            'valid/loss': loss.item(),
            'valid/step': step,
        })
        return loss

    def train_loop(self):
        self.model.train()
        prog_bar = tqdm(total=self.config.max_steps)
        num_steps = self.config.max_steps
        step = 0
        while True:
            for batch in self.train_loader:
                step += 1
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
            if step >= num_steps:
                prog_bar.set_description('Training Finished')
                break

            self.train_loader = iter(self.train_dataset)

    def valid_loop(self):
        self.model.eval()
        prog_bar = tqdm(total=self.config.max_valid_steps)
        step = 0

        for index, batch in enumerate(self.valid_loader):
            step += 1
            valid_loss = self.valid_step(batch, step + 1 + self.cur_valid_step)
            prog_bar.update(1)
            if step % self.config.log_intervals == 0:
                log_info = {
                    'mode': 'VALIDATION',
                    'loss': f"{valid_loss:.4f}"
                }
                prog_bar.set_postfix(log_info)

            if index >= self.config.max_valid_steps:
                break

        self.cur_valid_step += self.config.max_valid_steps
        self.model.train()


    @torch.no_grad()
    def save_ckpt(self, step):
        os.makedirs(self.config.ckpt_dir, exist_ok=True)

        weights = self.model.state_dict()
        filename = os.path.join(self.config.ckpt_dir, f'weights-{step}.safetensors')
        save_file(weights, filename)

        ema_weights = self.ema_model.state_dict()
        ema_filename = os.path.join(self.config.ckpt_dir, f'ema-weights-{step}.safetensors')
        save_file(ema_weights, ema_filename)

        if self.config.save_plots:
            plots_dir = self.config.save_plots
            image_dir = os.path.join(plots_dir, f'flow-map-plots-{step}')
            os.makedirs(image_dir, exist_ok=True)

            results = self.ema_model.sample()
            x_hat = results[-1].detach().float().cpu()

            batch = next(self.valid_loader)
            x_true = batch[0][:1].detach().float().cpu()

            save_image(denormalize(x_hat), os.path.join(image_dir, "end.png"))
            save_image(denormalize(results[0].detach().float().cpu()), os.path.join(image_dir, "start.png"))
            save_image(denormalize(x_true.detach()), os.path.join(image_dir, "true.png"))

            if self.use_wandb:
                wandb.log({
                    "samples/start": wandb.Image(os.path.join(image_dir, "start.png")),
                    "samples/end": wandb.Image(os.path.join(image_dir, "end.png")),
                    "samples/true": wandb.Image(os.path.join(image_dir, "true.png")),
                })
            else:
                print(f"Saved sample images to {image_dir}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    config = CifarConfig()
    model = Unet().to(device)
    trainer = FlowTrainer(
        model = model, config = config, device = device
    )
    trainer.train()
