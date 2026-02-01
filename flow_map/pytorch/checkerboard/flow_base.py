import os
from dataclasses import dataclass, asdict
import torch
from torch.nn import functional as F
from torch import nn, Tensor, optim
from flow_map.pytorch.checkerboard.data import get_loader as checkerboard_loader
from flow_map.trainer import Config, BaseTrainer
from flow_map.common import plot_checkerboard
from safetensors.torch import save_file
import wandb
from transformers import get_linear_schedule_with_warmup


@dataclass
class CheckersConfig(Config):
    n_hidden: int = 4
    n_neurons: int = 512
    output_dim: int = 2

    checkers_n_samples: int = 3000

    max_steps: int = 10000
    batch_size: int = 2
    lr: float = 1e-3
    warmup_ratio: float = 0.2
    max_valid_steps: int = 10
    valid_interval: int = 1000

    # torch.compile settings
    use_compile: bool = True
    compile_mode: str = "max-autotune"  # "default", "reduce-overhead", "max-autotune"
    compile_fullgraph: bool = True  # enforce no graph breaks
    compile_dynamic: bool = False  # static shapes for MLP


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


def maybe_compile(model: nn.Module, config: "CheckersConfig", device: torch.device) -> nn.Module:
    if not config.use_compile:
        print("torch.compile disabled in config")
        return model

    if device.type != "cuda":
        print(f"torch.compile skipped: device is {device.type}, not cuda")
        return model

    if not hasattr(torch, "compile"):
        print("torch.compile not available (requires PyTorch 2.0+)")
        return model

    print(f"Compiling model with mode={config.compile_mode}, "
          f"fullgraph={config.compile_fullgraph}, dynamic={config.compile_dynamic}")

    compiled = torch.compile(
        model,
        mode=config.compile_mode,
        fullgraph=config.compile_fullgraph,
        dynamic=config.compile_dynamic,
    )
    return compiled


class FlowTrainer(BaseTrainer):
    def __init__(self,
                 model: nn.Module,
                 config: CheckersConfig,
                 device=None
                 ):
        super().__init__(model, config)

        self.device = device
        self.model = self.model.to(device)
        self.model = maybe_compile(self.model, config, device)

        self.configure_optimizers()
        self.configure_loaders()

        wandb.init(project='flow maps', entity='jenrola2292', config=
        asdict(config))

    def configure_optimizers(self):
        optimizer = optim.RAdam(
            self.model.parameters(),
            lr=self.config.lr
        )
        warmup_steps = int(self.config.max_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=self.config.max_steps
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_loaders(self):
        loader = checkerboard_loader(
            batch_size=self.config.batch_size,
            n_samples=self.config.checkers_n_samples,
            max_width=4,
            max_height=4,
        )
        loader = iter(loader)
        self.train_loader = loader
        self.valid_loader = loader

    def compute_loss(self, data):
        x_1, width, height = data
        x_1 = x_1.to(self.device)

        batch_size = x_1.size(0)
        x_0 = torch.randn_like(x_1, device=self.device)
        t = torch.rand([batch_size], device=self.device)
        x_t = self._get_interpolant(x_0, x_1, t)
        flow = x_1 - x_0
        pred = self.model(x_t, t)
        loss = F.mse_loss(pred, flow)
        return loss

    def _get_interpolant(self, x_0: Tensor, x_1: Tensor, t: Tensor):
        # Use view instead of unsqueeze for compile compatibility
        t = t.view(-1, 1, 1)
        x_t = (1 - t) * x_0 + t * x_1
        return x_t

    def save_ckpt(self, step):
        # Get underlying model if compiled
        model_to_save = self.model
        if hasattr(self.model, "_orig_mod"):
            model_to_save = self.model._orig_mod

        weights = model_to_save.state_dict()
        os.makedirs(self.config.ckpt_dir, exist_ok=True)
        filename = os.path.join(self.config.ckpt_dir, f'weights-{step}.safetensors')
        save_file(weights, filename)

        if self.config.save_plots:
            plots_dir = self.config.save_plots
            image_dir = os.path.join(plots_dir, f'flow-map-plots-{step}')
            os.makedirs(image_dir, exist_ok=True)
            start_path = os.path.join(image_dir, 'start.png')
            end_path = os.path.join(image_dir, 'end.png')
            # Use the underlying model for sampling to avoid recompilation
            results = model_to_save.sample(self.config.checkers_n_samples)
            x_hat = results[-1]
            x_true = next(self.train_loader)[0][0].to(self.device)
            w1 = sliced_wasserstein(x_hat, x_true)
            wandb.log({'w1': w1.item()})
            plot_checkerboard(results[0].cpu().detach().numpy(), save=start_path)
            plot_checkerboard(results[-1].cpu().detach().numpy(), save=end_path)

    def train_step(self, batch, step):
        loss = self.compute_loss(batch)
        lr = self.optimizer.param_groups[0]['lr']

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        wandb.log({
            'train/loss': loss.item(),
            'global/lr': lr,
            'train/step': step
        })
        return loss

    def valid_step(self, batch, step):
        loss = self.compute_loss(batch)
        wandb.log({
            'valid/loss': loss.item(),
            'valid/step': step
        })
        return loss


class CondFlowTrainer(BaseTrainer):
    def __init__(self,
                 model: nn.Module,
                 config: CheckersConfig,
                 device=None
                 ):
        super().__init__(model, config)

        self.device = device
        self.model = self.model.to(device)
        self.model = maybe_compile(self.model, config, device)

        self.configure_optimizers()
        self.configure_loaders()

        wandb.init(project='flow maps', entity='jenrola2292', config=
        asdict(config))

    def configure_optimizers(self):
        optimizer = optim.RAdam(
            self.model.parameters(),
            lr=self.config.lr
        )
        warmup_steps = int(self.config.max_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=self.config.max_steps
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_loaders(self):
        loader = checkerboard_loader(
            batch_size=self.config.batch_size,
            n_samples=self.config.checkers_n_samples,
            max_width=5,
            max_height=5,
            cond=True
        )
        loader = iter(loader)
        self.train_loader = loader
        self.valid_loader = loader

    def compute_loss(self, data):
        x_1, width, height = data
        x_1 = x_1.to(self.device)
        width = width.to(self.device)
        height = height.to(self.device)
        c = width * height

        batch_size = x_1.size(0)
        x_0 = torch.randn_like(x_1, device=self.device)
        t = torch.rand([batch_size], device=self.device)
        x_t = self._get_interpolant(x_0, x_1, t)
        flow = x_1 - x_0
        pred = self.model(x_t, t, c)
        loss = F.mse_loss(pred, flow)
        return loss

    def _get_interpolant(self, x_0: Tensor, x_1: Tensor, t: Tensor):
        t = t.view(-1, 1, 1)
        x_t = (1 - t) * x_0 + t * x_1
        return x_t

    def save_ckpt(self, step):
        model_to_save = self.model
        if hasattr(self.model, "_orig_mod"):
            model_to_save = self.model._orig_mod

        weights = model_to_save.state_dict()
        os.makedirs(self.config.ckpt_dir, exist_ok=True)
        filename = os.path.join(self.config.ckpt_dir, f'weights-{step}.safetensors')
        save_file(weights, filename)

        if self.config.save_plots:
            plots_dir = self.config.save_plots
            image_dir = os.path.join(plots_dir, f'flow-map-plots-{step}')
            os.makedirs(image_dir, exist_ok=True)
            start_path = os.path.join(image_dir, 'start.png')
            end_path = os.path.join(image_dir, 'end.png')
            results = model_to_save.sample(self.config.checkers_n_samples)
            x_hat = results[-1]
            x_true = next(self.train_loader)[0][0].to(self.device)
            w1 = sliced_wasserstein(x_hat, x_true)
            wandb.log({'w1': w1.item()})
            plot_checkerboard(results[0].cpu().detach().numpy(), save=start_path)
            plot_checkerboard(results[-1].cpu().detach().numpy(), save=end_path)

    def train_step(self, batch, step):
        loss = self.compute_loss(batch)
        lr = self.optimizer.param_groups[0]['lr']

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        wandb.log({
            'train/loss': loss.item(),
            'global/lr': lr,
            'train/step': step
        })
        return loss

    def valid_step(self, batch, step):
        loss = self.compute_loss(batch)
        wandb.log({
            'valid/loss': loss.item(),
            'valid/step': step
        })
        return loss


def train_flow():
    from flow_map.pytorch.checkerboard.flow_model import MLP

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')
    model = MLP()
    config = CheckersConfig()
    trainer = FlowTrainer(model, config, device=device)

    trainer.train()


def train_cond_flow():
    from flow_map.pytorch.checkerboard.flow_model import CondMLP

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')
    model = CondMLP()
    config = CheckersConfig()
    trainer = CondFlowTrainer(model, config, device=device)

    trainer.train()


if __name__ == "__main__":
    train_cond_flow()