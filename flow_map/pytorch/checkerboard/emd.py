import os
from dataclasses import dataclass
import torch
from torch import nn, Tensor, optim
from flow_map.pytorch.data import checkerboard_loader
from flow_map.trainer import Config, BaseTrainer
from flow_map.pytorch.checkerboard.flow_base import sliced_wasserstein
from flow_map.common import plot_checkerboard
from safetensors.torch import save_file


@dataclass
class CheckersConfig(Config):
    n_hidden:int = 4
    n_neurons:int = 512
    output_dim:int = 2

    checkers_n_samples:int = 2000

    max_steps:int = 50000
    batch_size:int = 8
    lr:float = 1e-3
    warmup_ratio:float = 0.2
    max_valid_steps:int = 10
    valid_interval:int = 50

class EMDTrainer(BaseTrainer):
    def __init__(
            self,
            teacher:nn.Module,
            student:nn.Module,
            config:Config,
            device = None
    ):
        super().__init__(
            student, config
        )
        self.device = device
        self.teacher = teacher.to(self.device).eval()
        self.model = self.model.to(self.device)

        self.configure_optimizers()
        self.configure_loaders()
        #self.exp = start(project_name='flow-maps',workspace='odunola')
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
            max_width=4,
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
            results = self.model.sample(self.config.checkers_n_samples)
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

    def run_student_model(self, I, s, t):
        v_s_t = self.model(
            x=I, s=s, t=t
        )
        return v_s_t

    def compute_eulerian_residual(self,
                                  I:Tensor, start_time:Tensor, end_time:Tensor ):
        v_ss = self.teacher(I, start_time)

        def func(s, x):
            dt = end_time - s
            return x + (dt[:, None, None] * self.run_student_model(x, s, end_time))

        X_t, D_X_t = torch.func.jvp(
            func, primals = (start_time, I), tangents=(
                torch.ones_like(start_time), v_ss.detach()
            )
        )
        return X_t, D_X_t


    def compute_loss(self, data):
        x_1, width, height = data
        x_1  =x_1.to(self.device)

        batch_size = x_1.size(0)
        x_0 = torch.randn_like(x_1, device = self.device)
        s = torch.rand([batch_size], device = self.device)
        t = torch.rand([batch_size], device = self.device)
        t = s + t * (1-s)

        I_s = self.get_interpolant(x_0, x_1, s)
        X_t, residual = self.compute_eulerian_residual(I_s, s, t)
        loss = residual.pow(2).mean()
        return loss


    def train_step(self, batch, step):
        loss = self.compute_loss(batch)
        lr = self.optimizer.param_groups[0]['lr']
        #self.exp.log_metrics({'train_loss': loss, 'lr': lr, 'train_step': step})
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss

    def valid_step(self, batch, step):
        loss = self.compute_loss(batch)
        #self.exp.log_metrics({'valid_loss': loss, 'valid_step': step})
        return loss



if __name__ == "__main__":
    from flow_map.pytorch.checkerboard.model import MLP, FlowMapMLP
    device = torch.device('mps')

    config = CheckersConfig()
    teacher = MLP()
    student = FlowMapMLP()
    loader = checkerboard_loader(
        batch_size=8,n_samples = 2000
    )
    batch = next(iter(loader))
    trainer = EMDTrainer(teacher, student, config, device=device)
    print(trainer.compute_loss(batch))
    trainer.train()



