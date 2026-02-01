import torch
from torch import nn, Tensor
from safetensors.torch import safe_open
import random


def euler(func, y0, t, c=None):
    x = y0.clone()
    xs = [y0]
    for i in range(len(t) - 1):
        t_i = t[i:i+1]
        t_next = t[i + 1]
        dt = t_next - t_i

        if c is not None:
            dx = func(x, t_i.squeeze(-1), c)
        else:
            dx = func(x, t_i.squeeze(-1))

        x = x + dt * dx
        xs.append(x.clone().squeeze(0))
    return xs


def sinusoidal_t(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, device=t.device) * torch.log(torch.tensor(10000.0)) / (half - 1)
    )
    emb = t * freqs
    return torch.cat([emb.sin(), emb.cos()], dim=-1)


class MLP(nn.Module):
    def __init__(
            self,
            n_neurons=512,
            output_dim=2
    ):
        super().__init__()

        self.time_proj = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            nn.GELU(),
        )

        layers = []
        layers.append(nn.Linear(output_dim, n_neurons))
        layers.append(nn.GELU())
        layers.append(nn.Linear(n_neurons, n_neurons))
        layers.append(nn.GELU())
        layers.append(nn.Linear(n_neurons, n_neurons))
        layers.append(nn.GELU())
        layers.append(nn.Linear(n_neurons, output_dim))

        self.layers = nn.ModuleList(layers)
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

        self.n_neurons = n_neurons

        #self.load_weights()

    def forward(self, x: Tensor, t: Tensor):
        t = t[:, None]
        t_embed = sinusoidal_t(t, self.n_neurons)
        t_embed = self.time_proj(t_embed)
        t_embed = t_embed[:, None, :]

        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
            if isinstance(layer, nn.Linear):
                x = (layer(x) * t_embed) + t_embed
            else:
                x = layer(x)
        x = self.layers[-1](x)
        return x

    def load_weights(self):
        weights = {}
        with safe_open(
                '/flow_map/pytorch/models/pt_flow.safetensors',
                'pt') as fp:
            for key in fp.keys():
                weights[key] = fp.get_tensor(key)

        self.load_state_dict(weights)

    def sample(self, num_points, num_steps: int = 100):
        device = next(self.parameters()).device
        y0 = torch.randn(1, num_points, 2, device=device)
        t = torch.linspace(0, 1, num_steps + 1,
                           device=device, dtype=torch.float32)
        t = t.unsqueeze(-1)
        y1 = euler(self, y0, t)
        return y1

class CondMLP(nn.Module):
    def __init__(
            self,
            n_neurons=384,
            output_dim=2
    ):
        super().__init__()

        self.time_proj = nn.Sequential(
            nn.Linear(n_neurons, n_neurons * 2),
            nn.GELU(),
            nn.Linear(n_neurons*2, n_neurons*2),
        )
        self.c_proj = nn.Sequential(
            nn.Embedding(30, n_neurons),  # hardcoding num classes for now
            nn.Linear(n_neurons, n_neurons *2),
            nn.GELU(),
            nn.Linear(n_neurons*2, n_neurons*2),
        )

        layers = []
        layers.append(nn.Linear(output_dim, n_neurons))
        layers.append(nn.GELU())
        layers.append(nn.Linear(n_neurons, n_neurons))
        layers.append(nn.GELU())
        layers.append(nn.Linear(n_neurons, n_neurons))
        layers.append(nn.GELU())
        layers.append(nn.Linear(n_neurons, output_dim))

        self.layers = nn.ModuleList(layers)
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

        self.n_neurons = n_neurons

        #self.load_weights()

    def forward(self, x: Tensor, t: Tensor, c: Tensor | tuple[Tensor, Tensor]):
        t = t[:, None]
        c = c[:, None]
        t_embed = sinusoidal_t(t, self.n_neurons)
        t_embed = self.time_proj(t_embed)
        t_embed = t_embed[:, None, :]
        t_scale, t_shift = torch.chunk(t_embed, 2, dim= -1)

        c_embed = self.c_proj(c)[:, None, :]
        c_scale, c_shift = torch.chunk(c_embed, 2, dim=-1)

        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
            if isinstance(layer, nn.Linear):
                x = (x * t_scale) + t_shift
                x = layer(x)
                x = (x * c_scale) + c_shift
            else:
                x = layer(x)
        x = self.layers[-1](x)
        return x

    def load_weights(self):
        weights = {}
        with safe_open(
                '/flow_map/pytorch/models/pt_flow.safetensors',
                'pt') as fp:
            for key in fp.keys():
                weights[key] = fp.get_tensor(key)

        self.load_state_dict(weights)

    def sample(self, num_points, num_classes = 5,num_steps: int = 100):
        device = next(self.parameters()).device
        y0 = torch.randn(1, num_points, 2, device=device)
        t = torch.linspace(0, 1, num_steps + 1,
                           device=device, dtype=torch.float32)
        t = t.unsqueeze(-1)
        c = torch.randint(0, num_classes, (num_classes,), device=device)
        y1 = euler(self, y0, t, c)
        return y1

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP()
    print(model)