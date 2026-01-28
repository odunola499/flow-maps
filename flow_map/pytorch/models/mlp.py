import torch
from torch import nn, Tensor
from dataclasses import dataclass

@dataclass
class Config:
    n_hidden:int
    n_neurons:int
    output_dim:int

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

class MLP(nn.Module):
    def __init__(
            self,
            n_neurons = 512,
            output_dim = 2
    ):
        super().__init__()

        self.time_proj = nn.Sequential(
            nn.Linear(1, n_neurons),
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

    def forward(self, x:Tensor, t:Tensor, c:Tensor|tuple[Tensor, Tensor] = None):
        t = t.unsqueeze(1).unsqueeze(1)

        t_embed = self.time_proj(t)

        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
            if isinstance(layer, nn.Linear):
                x = layer(x) + t_embed
            else:
                x = layer(x)
        x = self.layers[-1](x)
        return x

    def sample(self, num_points, num_steps:int = 100):
        device = next(self.parameters()).device
        y0 = torch.randn(1, num_points, 2, device = device)
        t = torch.linspace(0, 1, num_steps + 1, device = device, dtype = torch.float32)
        t = t.unsqueeze(-1)
        y1 = euler(self, y0, t)
        return y1

if __name__ == "__main__":

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    model = MLP().to(device)
    tensor = torch.randn(2, 100, 2, device=device)
    time = torch.rand([2], device=device)
    output = model(tensor, time)
    print(output.shape)