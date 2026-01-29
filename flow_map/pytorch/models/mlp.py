import torch
from torch import nn, Tensor
from dataclasses import dataclass
from safetensors.torch import safe_open


@dataclass
class Config:
    n_hidden: int
    n_neurons: int
    output_dim: int


def euler(func, y0, t):
    x = y0.clone()
    xs = [y0]
    for i in range(len(t) - 1):
        t_i = t[i]
        t_next = t[i + 1]
        dt = t_next - t_i

        dx = func(x, t_i)
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
            nn.Linear(n_neurons, n_neurons),
        )

        layers = []
        layers.append(nn.Linear(output_dim, n_neurons))
        layers.append(nn.GELU()),
        layers.append(nn.Linear(n_neurons, n_neurons))
        layers.append(nn.GELU()),
        layers.append(nn.Linear(n_neurons, n_neurons))
        layers.append(nn.GELU()),
        layers.append(nn.Linear(n_neurons, output_dim))

        self.layers = nn.ModuleList(layers)
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

        self.n_neurons = n_neurons

        self.load_weights()

    def forward(self, x: Tensor, t: Tensor, c: Tensor | tuple[Tensor, Tensor] = None):
        t = t[:, None]
        t_embed = sinusoidal_t(t, self.n_neurons)
        t_embed = self.time_proj(t_embed)
        t_embed = t_embed[:, None, :]

        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
            if isinstance(layer, nn.Linear):
                x = layer(x) + t_embed
            else:
                x = layer(x)
        x = self.layers[-1](x)
        return x

    def load_weights(self):
        weights = {}
        with safe_open(
                '/Users/odunolajenrola/Documents/GitHub/flow-maps/flow_map/pytorch/models/pt_flow.safetensors',
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


class FlowMapMLP(nn.Module):
    def __init__(self, n_neurons=256, output_dim=2):
        super().__init__()

        self.time_proj = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            nn.GELU(),
            nn.Linear(n_neurons, n_neurons),
        )
        # self.s_proj = nn.Sequential(
        #     nn.Linear(n_neurons, n_neurons),
        #     nn.GELU(),
        #     nn.Linear(n_neurons, n_neurons),
        # )

        layers = []
        layers.append(nn.Linear(output_dim, n_neurons))
        layers.append(nn.GELU()),
        layers.append(nn.Linear(n_neurons, n_neurons))
        layers.append(nn.GELU()),
        layers.append(nn.Linear(n_neurons, n_neurons))
        layers.append(nn.GELU()),
        layers.append(nn.Linear(n_neurons, output_dim))

        self.layers = nn.ModuleList(layers)
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

        self.n_neurons = n_neurons

    def forward(
            self,
            x:Tensor,
            s:Tensor,
            t:Tensor
    ):
        t = t[:, None]
        s = s[:, None]

        t = self.time_proj(sinusoidal_t(t, self.n_neurons))[:, None, :]
        s = self.time_proj(sinusoidal_t(s, self.n_neurons))[:, None, :]

        embed = (s * 0.5) + (t*0.5)
        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
            if isinstance(layer, nn.Linear):
                x = layer(x) + embed
            else:
                x = layer(x)
        x = self.layers[-1](x)
        return x

    def sample(self, num_points, num_steps:int = 3):
        device = next(self.parameters()).device
        y0 = torch.randn(1, num_points, 2, device=device)
        times = torch.linspace(0, 1, num_steps +1,
                               device=device, dtype=torch.float32)
        times = times.unsqueeze(-1)
        y = y0
        ys = [y]
        for step in range(num_steps):
            s = times[step]
            t = times[step + 1]
            dt = t - s
            velocity = self(y0, s, t)
            y += (dt * velocity)
            ys.append(y.clone().squeeze(0))
        return ys


def test_pretrained_model():
    from flow_map.common import plot_checkerboard

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = MLP().to(device)

    tensor = torch.randn(2, 100, 2, device=device)
    time = torch.rand([2], device=device)
    output = model(tensor, time)
    print(output.shape)

    results = model.sample(2000)
    end = results[-1].cpu().detach().numpy()
    plot_checkerboard(end, save=None)


if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = MLP().to(device)
    tensor = torch.randn(2, 100, 2, device=device)
    time = torch.rand([2], device=device)

    output = model(tensor, time)
    print(output)

    test_pretrained_model()
