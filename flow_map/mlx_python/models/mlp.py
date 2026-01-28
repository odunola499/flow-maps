import mlx.core as mx
import mlx.nn as nn


class MLP(nn.Module):
    def __init__(
            self,
            n_hidden = 4,
            n_neurons = 512,
            output_dim = 2
    ):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons
        self.output_dim = output_dim

        self.act = nn.GELU()
        self.input_proj = nn.Linear(output_dim, n_neurons)

        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(n_neurons, n_neurons))
        self.intermediate_proj = nn.Sequential(*layers)

        self.output_proj = nn.Linear(n_neurons, output_dim)

    def __call__(self, x:mx.array):
        x = self.input_proj(x)
        x = self.intermediate_proj(x)
        x = self.output_proj(x)
        return x

if __name__ == "__main__":
    model = MLP()
    device = mx.default_device()

    tensor = mx.random.normal((2, 100,2))
    output = model(tensor)
    print(output.shape)