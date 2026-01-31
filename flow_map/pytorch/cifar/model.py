import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


def sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, device=t.device, dtype=t.dtype) *
        math.log(10000.0) / (half - 1)
    )
    args = t[:, None] * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)

        q = self.q(h).view(B, C, -1)
        k = self.k(h).view(B, C, -1)
        v = self.v(h).view(B, C, -1)

        attn = torch.bmm(q.permute(0, 2, 1), k) * self.scale
        attn = F.softmax(attn, dim=-1)

        h = torch.bmm(v, attn.permute(0, 2, 1))
        h = h.view(B, C, H, W)
        return x + self.out(h)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mults: tuple = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attn_resolutions: tuple = (16,),
        dropout: float = 0.1,
        num_classes: int = 10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_classes = num_classes

        emb_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.class_emb = nn.Embedding(num_classes, emb_dim)

        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = [base_channels]
        now_ch = base_channels
        resolution = 32

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult

            for _ in range(num_res_blocks):
                layers = nn.ModuleList([ResBlock(now_ch, out_ch, emb_dim, dropout)])
                now_ch = out_ch
                if resolution in attn_resolutions:
                    layers.append(AttnBlock(now_ch))
                self.downs.append(layers)
                channels.append(now_ch)

            if level < len(channel_mults) - 1:
                self.downs.append(nn.ModuleList([Downsample(now_ch)]))
                channels.append(now_ch)
                resolution //= 2

        self.mid = nn.ModuleList([
            ResBlock(now_ch, now_ch, emb_dim, dropout),
            AttnBlock(now_ch),
            ResBlock(now_ch, now_ch, emb_dim, dropout),
        ])

        for level, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult

            for i in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                layers = nn.ModuleList([ResBlock(now_ch + skip_ch, out_ch, emb_dim, dropout)])
                now_ch = out_ch
                if resolution in attn_resolutions:
                    layers.append(AttnBlock(now_ch))
                self.ups.append(layers)

            if level < len(channel_mults) - 1:
                self.ups.append(nn.ModuleList([Upsample(now_ch)]))
                resolution *= 2

        self.out_norm = nn.GroupNorm(min(32, now_ch), now_ch)
        self.out_conv = nn.Conv2d(now_ch, out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: Tensor, t: Tensor, c: Tensor) -> Tensor:
        t_emb = sinusoidal_embedding(t, self.base_channels)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.class_emb(c)
        emb = t_emb + c_emb

        h = self.input_conv(x)
        hs = [h]

        for module in self.downs:
            for layer in module:
                if isinstance(layer, ResBlock):
                    h = layer(h, emb)
                elif isinstance(layer, Downsample):
                    h = layer(h)
                else:
                    h = layer(h)
            hs.append(h)

        for layer in self.mid:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
            else:
                h = layer(h)

        for module in self.ups:
            if isinstance(module[0], Upsample):
                h = module[0](h)
            else:
                h = torch.cat([h, hs.pop()], dim=1)
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, emb)
                    else:
                        h = layer(h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


class CondFlowMapUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mults: tuple = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attn_resolutions: tuple = (16,),
        dropout: float = 0.1,
        num_classes: int = 10,
    ):
        super().__init__()
        self.unet = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            num_classes=num_classes,
        )
        self.num_classes = num_classes
        self.base_channels = base_channels

    def forward(self, x: Tensor, s: Tensor, t: Tensor, c: Tensor) -> Tensor:
        combined_t = t - s
        return self.unet(x, combined_t, c)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        num_steps: int = 4,
        c: Tensor = None,
        device: torch.device = None,
    ) -> Tensor:
        if device is None:
            device = next(self.parameters()).device

        if c is None:
            c = torch.randint(0, self.num_classes, (batch_size,), device=device)

        x = torch.randn(batch_size, 3, 32, 32, device=device)
        times = torch.linspace(0, 1, num_steps + 1, device=device)

        for i in range(num_steps):
            s = times[i].expand(batch_size)
            t = times[i + 1].expand(batch_size)
            dt = t - s
            velocity = self(x, s, t, c)
            x = x + dt[:, None, None, None] * velocity

        return x


if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    model = CondFlowMapUNet(base_channels=64).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    x = torch.randn(4, 3, 32, 32, device=device)
    s = torch.rand(4, device=device)
    t = torch.rand(4, device=device)
    c = torch.randint(0, 10, (4,), device=device)

    out = model(x, s, t, c)
    print(f"Output shape: {out.shape}")

    samples = model.sample(4, num_steps=4)
    print(f"Sample shape: {samples.shape}")
