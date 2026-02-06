import torch
from torch import nn, Tensor
from safetensors.torch import safe_open
import math
from torch.nn import functional as F

LOG_10000 = math.log(10000.0)


def euler(func, y0, t, c=None):
    x = y0.clone()
    xs = [y0]
    for i in range(len(t) - 1):
        t_i = t[i:i + 1]
        t_next = t[i + 1]
        dt = t_next - t_i
        batch = x.size(0)
        t_batch = t_i.squeeze(-1).expand(batch)

        if c is not None:
            dx = func(x, t_batch, c)
        else:
            dx = func(x, t_batch)

        x = x + dt * dx
    xs.append(x.clone().squeeze(0))
    return xs


def sinusoidal_t(t: Tensor, dim: int) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, device=t.device, dtype=t.dtype) * (LOG_10000 / (half - 1))
    )
    emb = t * freqs
    return torch.cat([emb.sin(), emb.cos()], dim=-1)


class MLPBlock(nn.Module):
    def __init__(self, n_neurons: int):
        super().__init__()
        self.linear = nn.Linear(n_neurons, n_neurons)
        self.act = nn.GELU()

    def forward(self, x: Tensor, t_embed: Tensor) -> Tensor:
        x = (self.linear(x) * t_embed) + t_embed
        x = self.act(x)
        return x


class MLP(nn.Module):
    def __init__(self, n_neurons: int = 512, output_dim: int = 2):
        super().__init__()
        self.n_neurons = n_neurons

        self.time_proj = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            nn.GELU(),
        )

        self.input_proj = nn.Linear(output_dim, n_neurons)
        self.input_act = nn.GELU()

        self.block1 = MLPBlock(n_neurons)
        self.block2 = MLPBlock(n_neurons)
        self.block3 = MLPBlock(n_neurons)
        self.block4 = MLPBlock(n_neurons)

        self.output_proj = nn.Linear(n_neurons, output_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t = t.view(-1, 1)
        t_embed = sinusoidal_t(t, self.n_neurons)
        t_embed = self.time_proj(t_embed)
        t_embed = t_embed.unsqueeze(1)

        x = self.input_proj(x)
        x = self.input_act(x)

        x = self.block1(x, t_embed)
        x = self.block2(x, t_embed)
        x = self.block3(x, t_embed)
        x = self.block4(x, t_embed)

        x = self.output_proj(x)
        return x

    def sample(self, num_points: int, num_steps: int = 100):
        device = next(self.parameters()).device
        y0 = torch.randn(1, num_points, 2, device=device)
        t = torch.linspace(0, 1, num_steps + 1, device=device, dtype=torch.float32)
        t = t.unsqueeze(-1)
        y1 = euler(self, y0, t)
        return y1


class CondBlock(nn.Module):
    def __init__(self, n_neurons: int):
        super().__init__()
        hidden = 4 * n_neurons
        self.norm = nn.LayerNorm(n_neurons, elementwise_affine=False)
        self.w1 = nn.Linear(n_neurons, hidden, bias=False)
        self.w2 = nn.Linear(hidden, n_neurons, bias=False)
        self.w3 = nn.Linear(n_neurons, hidden, bias=False)

    def forward(self, x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
        h = self.norm(x)
        h = h * (1 + scale) + shift
        h = self.w2(F.silu(self.w1(h)) * self.w3(h))
        return x + h  # residual


class CondMLP(nn.Module):
    def __init__(self, n_neurons: int = 512, output_dim: int = 2, num_classes: int = 40):
        super().__init__()
        self.n_neurons = n_neurons

        self.time_proj = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            nn.SiLU(),
            nn.Linear(n_neurons, n_neurons),
        )
        self.c_embed = nn.Embedding(num_classes, n_neurons)

        self.input_proj = nn.Linear(output_dim, n_neurons)

        self.blocks = nn.ModuleList([CondBlock(n_neurons) for _ in range(2)])

        self.cond_proj = nn.Linear(n_neurons, n_neurons * 2 * 2)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

        self.final_norm = nn.LayerNorm(n_neurons, elementwise_affine=False)
        self.output_proj = nn.Linear(n_neurons, output_dim)

        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self.load_weights()

    def forward(self, x: Tensor, t: Tensor, c: Tensor) -> Tensor:
        t_embed = sinusoidal_t(t.view(-1, 1), self.n_neurons)
        t_embed = self.time_proj(t_embed)
        c_embed = self.c_embed(c)
        cond = t_embed + c_embed

        mods = self.cond_proj(cond).unsqueeze(1).chunk(4, dim=-1)

        x = self.input_proj(x)
        for i, block in enumerate(self.blocks):
            x = block(x, mods[i * 2], mods[i * 2 + 1])

        x = self.final_norm(x)
        x = self.output_proj(x)
        return x

    def load_weights(self):
        weights = {}
        with safe_open(
                '/root/flow-maps/flow_map/pytorch/checkerboard/checkpoints/weights-1000.safetensors',
                'pt') as fp:
            for key in fp.keys():
                weights[key] = fp.get_tensor(key)

        print('loaded weights')
        self.load_state_dict(weights)

    @torch.no_grad()
    def sample(self, num_points: int, c_value: int = 15, num_steps: int = 100):
        device = next(self.parameters()).device
        y0 = torch.randn(1, num_points, 2, device=device)
        t = torch.linspace(0, 1, num_steps + 1, device=device, dtype=torch.float32)
        t = t.unsqueeze(-1)
        c = torch.tensor([c_value], device=device, dtype=torch.long)
        y1 = euler(self, y0, t, c)
        return y1


if __name__ == "__main__":
    from flow_map.common import plot_checkerboard

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CondMLP()
    print(model)
    model.load_weights()
    results = model.sample(3000)
    plot_checkerboard(results[-1].cpu().detach().numpy(), save='test.png')