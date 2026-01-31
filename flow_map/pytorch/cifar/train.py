import os
import math
import copy
import glob
import shutil
from dataclasses import dataclass
import torch
from torch import Tensor, optim
from torch.amp import GradScaler
from torchvision.utils import save_image

from flow_map.trainer import Config, BaseTrainer
from flow_map.pytorch.cifar.data import CIFAR10Dataset
from flow_map.pytorch.cifar.model import CondFlowMapUNet
from safetensors.torch import save_file
from transformers import get_cosine_schedule_with_warmup


@dataclass
class CIFARConfig(Config):
    base_channels: int = 128
    channel_mults: tuple = (1, 2, 2, 2)
    num_res_blocks: int = 2
    attn_resolutions: tuple = (16,)
    dropout: float = 0.1
    num_classes: int = 10

    max_steps: int = 500000
    batch_size: int = 4
    lr: float = 2e-4
    warmup_ratio: float = 0.01
    max_valid_steps: int = 10
    valid_interval: int = 5000

    ema_decay: float = 0.9999
    max_checkpoints: int = 5
    use_amp: bool = True

    data_root: str = "./data"
    num_workers: int = 4


@torch.no_grad()
def compute_fid_features(images: Tensor) -> Tensor:
    return images.view(images.size(0), -1).mean(dim=1)


class LSDTrainer(BaseTrainer):
    def __init__(self, model: CondFlowMapUNet, config: CIFARConfig, device, n: float = 0.75):
        super().__init__(model, config)
        self.device = device
        self.n = n
        self.config = config


        self.model: CondFlowMapUNet = self.model.to(device)

        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_decay = config.ema_decay

        self.scaler = GradScaler() if config.use_amp else None

    @torch.no_grad()
    def update_ema(self):
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            int(self.config.max_steps * self.config.warmup_ratio),
            self.config.max_steps,
        )

    def configure_loaders(self):
        dataset = CIFAR10Dataset(
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            data_root=self.config.data_root,
        )
        self.train_loader = dataset.get_train_iterator()
        self.valid_loader = dataset.get_valid_iterator()

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
        plot_dirs = glob.glob(os.path.join(plots_dir, 'samples-*'))

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
            image_dir = os.path.join(plots_dir, f'samples-{step}')
            os.makedirs(image_dir, exist_ok=True)

            for class_idx in range(min(10, self.config.num_classes)):
                c = torch.full((16,), class_idx, device=self.device, dtype=torch.long)
                samples = self.ema_model.sample(16, num_steps=4, c=c, device=self.device)
                samples = (samples + 1) / 2
                samples = samples.clamp(0, 1)
                save_path = os.path.join(image_dir, f'class_{class_idx}.png')
                save_image(samples, save_path, nrow=4)

            self._cleanup_old_plots()

    def get_interpolant(self, x_0: Tensor, x_1: Tensor, time: Tensor) -> Tensor:
        time = time[:, None, None, None]
        return (1 - time) * x_0 + time * x_1

    def compute_weight(self, s: Tensor, t: Tensor) -> Tensor:
        return torch.ones_like(s)

    def compute_time_derivative(self, I: Tensor, start_time: Tensor, end_time: Tensor, c: Tensor):
        def func(t):
            dt = t - start_time
            return I + (dt[:, None, None, None] * self.model(I, start_time, t, c))

        X_t, D_X_t = torch.func.jvp(
            func, (end_time,), (torch.ones_like(end_time),)
        )
        return X_t, D_X_t

    def compute_lsd_residual(self, x_1: Tensor, c: Tensor):
        B = x_1.shape[0]
        x_0 = torch.randn_like(x_1)

        temp1 = torch.rand([B], device=self.device)
        temp2 = torch.rand([B], device=self.device)
        s = torch.minimum(temp1, temp2)
        t = torch.maximum(temp1, temp2)

        I_s = self.get_interpolant(x_0, x_1, s)
        X_st, D_X_st = self.compute_time_derivative(I_s, s, t, c)

        v_tt = self.ema_model(X_st.detach(), t, t, c)

        error_sq = (D_X_st - v_tt).pow(2).mean(dim=[1, 2, 3])
        weight = self.compute_weight(s, t)
        weighted_loss = (torch.exp(-weight) * error_sq + weight).mean()

        return weighted_loss

    def compute_flow_residual(self, x_1: Tensor, c: Tensor):
        B = x_1.shape[0]
        x_0 = torch.randn_like(x_1)
        t = torch.rand([B], device=self.device)

        x_t = self.get_interpolant(x_0, x_1, t)
        flow = x_1 - x_0
        pred = self.model(x_t, t, t, c)

        error_sq = (pred - flow).pow(2).mean(dim=[1, 2, 3])
        weight = self.compute_weight(t, t)
        weighted_loss = (torch.exp(-weight) * error_sq + weight).mean()

        return weighted_loss

    def compute_loss(self, data):
        x_1, c = data
        x_1 = x_1.to(self.device)
        c = c.to(self.device)

        M = x_1.shape[0]
        Md = math.floor(self.n * M)
        if Md <= 0:
            Md = 1
        if Md >= M:
            Md = M - 1

        flow_loss = self.compute_flow_residual(x_1[:Md], c[:Md])
        lsd_loss = self.compute_lsd_residual(x_1[Md:], c[Md:])

        return flow_loss + lsd_loss, flow_loss, lsd_loss

    def train_step(self, batch, step):
        self.model.train()

        if self.scaler is not None:
            with torch.amp.autocast():
                loss, flow_loss, lsd_loss = self.compute_loss(batch)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss, flow_loss, lsd_loss = self.compute_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        self.scheduler.step()
        self.update_ema()

        if step % 100 == 0:
            print(f"Step {step}: loss={loss.item():.4f}, flow={flow_loss.item():.4f}, lsd={lsd_loss.item():.4f}")

        return loss

    def valid_step(self, batch, step):
        self.model.eval()
        with torch.no_grad():
            loss, flow_loss, lsd_loss = self.compute_loss(batch)
        return loss


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

    config = CIFARConfig(
        base_channels=64,
        batch_size=32,
        max_steps=100000,
        valid_interval=5000,
        use_amp=False,
    )

    model = CondFlowMapUNet(
        base_channels=config.base_channels,
        channel_mults=config.channel_mults,
        num_res_blocks=config.num_res_blocks,
        attn_resolutions=config.attn_resolutions,
        dropout=config.dropout,
        num_classes=config.num_classes,
    )

    trainer = LSDTrainer(model, config, device)
    trainer.train()
