import torch
from torch import nn, Tensor
from torch.nn import functional as F


def timestep_embedding(timesteps: Tensor, dim: int) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, device=timesteps.device, dtype=torch.float32)
        * (torch.log(torch.tensor(10000.0, device=timesteps.device)) / (half - 1))
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


@torch.no_grad()
def euler(func, y0: Tensor, t: Tensor, c: Tensor | None = None):
    if t.ndim == 2 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    x = y0.clone()
    xs = [x.clone()]
    for i in range(len(t) - 1):
        t_i = t[i]
        t_next = t[i + 1]
        dt = t_next - t_i
        t_batch = t_i.expand(x.size(0))
        dx = func(x, t_batch, c) if c is not None else func(x, t_batch)
        x = x + dt * dx
        xs.append(x.clone())
    return xs


class ConditionBlock(nn.Module):
    def __init__(self, emb_dim: int, channels: int):
        super().__init__()
        self.time_proj = nn.Linear(emb_dim, channels)
        self.film = nn.Linear(emb_dim, channels * 2)

    def forward(self, x: Tensor, t_emb: Tensor, y_emb: Tensor) -> Tensor:
        x = x + self.time_proj(t_emb)[:, :, None, None]
        gamma, beta = self.film(y_emb).chunk(2, dim=1)
        return gamma[:, :, None, None] * x + beta[:, :, None, None]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.cond = ConditionBlock(emb_dim, out_channels)
        self.skip = None if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor, t_emb: Tensor, y_emb: Tensor) -> Tensor:
        residual = x if self.skip is None else self.skip(x)
        h = self.conv1(x)
        h = self.cond(h, t_emb, y_emb)
        h = F.relu(h)
        h = self.conv2(h)
        return F.relu(h + residual)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(B, C, H * W).permute(0, 2, 1)
        k = k.view(B, C, H * W).permute(0, 2, 1)
        v = v.view(B, C, H * W).permute(0, 2, 1)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 2, 1).view(B, C, H, W)
        return x + self.proj(attn)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels, emb_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, emb_dim)
        self.down = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor, t_emb: Tensor, y_emb: Tensor) -> Tensor:
        x = self.res1(x, t_emb, y_emb)
        x = self.res2(x, t_emb, y_emb)
        return F.relu(self.down(x))


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, use_attn: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.attn = AttentionBlock(in_channels) if use_attn else nn.Identity()
        self.res1 = ResidualBlock(in_channels, in_channels, emb_dim)
        self.res2 = ResidualBlock(in_channels, in_channels, emb_dim)
        self.out = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: Tensor, t_emb: Tensor, y_emb: Tensor, skip: Tensor) -> Tensor:
        x = x + skip
        x = self.up(x)
        x = self.attn(x)
        x = self.res1(x, t_emb, y_emb)
        x = self.res2(x, t_emb, y_emb)
        return self.out(x)


class Unet(nn.Module):
    def __init__(self, num_blocks=6, base_channels=64, num_classes=10, emb_dim=256):
        super().__init__()

        if num_blocks % 2 != 0:
            num_blocks += 1

        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.class_embed = nn.Embedding(num_classes, emb_dim)

        channels = [3 if i == 0 else base_channels * (2 ** (i - 1))
                    for i in range((num_blocks // 2) + 1)]

        self.downs = nn.ModuleList([
            Downsample(channels[i], channels[i + 1], emb_dim)
            for i in range(num_blocks // 2)
        ])

        ups = []
        for i in range(num_blocks // 2):
            ups.append(Upsample(channels[i + 1], channels[i], emb_dim, use_attn=True))
        self.ups = nn.ModuleList(ups[::-1])

    def forward(self, x: Tensor, timestep: Tensor, y: Tensor | None = None):
        if y is None:
            y = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        t_emb = timestep_embedding(timestep, self.emb_dim)
        t_emb = self.time_mlp(t_emb)
        y_emb = self.class_embed(y)

        skips = []
        for down in self.downs:
            x = down(x, t_emb, y_emb)
            skips.append(x)

        for up in self.ups:
            x = up(x, t_emb, y_emb, skips.pop())

        return x

    @torch.no_grad()
    def sample(self, num_steps: int = 100, batch_size: int = 1, size: int = 32):
        device = next(self.parameters()).device
        y0 = torch.randn(batch_size, 3, size, size, device=device)
        t = torch.linspace(0, 1, num_steps + 1, device=device)
        c = torch.randint(0, 10, (batch_size,), device=device)
        return euler(self, y0, t, c)


def run_test():
    model = Unet()
    print(model)
    print(sum(p.numel() for p in model.parameters()))
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))
    y = torch.randint(0, 10, (2,))
    out = model(x, t, y)
    print(out.shape)


if __name__ == "__main__":
    run_test()
