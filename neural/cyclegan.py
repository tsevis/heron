"""CycleGAN — unpaired RGB <-> thermal translation (CLAUDE.md §2.2B).

Zhu et al. 2017. Learns visible->thermal without paired data, which is what makes
it practical: unpaired thermal imagery is far easier to obtain than pixel-aligned
pairs. Two generators (G: RGB->IR, F: IR->RGB) and two PatchGAN discriminators,
trained with adversarial + cycle-consistency + identity losses.

Runs on MPS. This is a *training* module: it needs a dataset (see
``neural/datasets.py``) and hours-to-days of compute — unlike the ComfyUI and
style-transfer finishers, it is not usable the moment it is written.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from heron.core.device import get_device


class ResnetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """9-block ResNet generator (the standard CycleGAN backbone)."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3, ngf: int = 64, n_blocks: int = 9):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_ch, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        for i in range(2):  # downsample
            m = 2 ** i
            layers += [nn.Conv2d(ngf * m, ngf * m * 2, 3, stride=2, padding=1),
                       nn.InstanceNorm2d(ngf * m * 2), nn.ReLU(True)]
        layers += [ResnetBlock(ngf * 4) for _ in range(n_blocks)]
        for i in range(2):  # upsample
            m = 2 ** (2 - i)
            layers += [nn.ConvTranspose2d(ngf * m, ngf * m // 2, 3, stride=2, padding=1, output_padding=1),
                       nn.InstanceNorm2d(ngf * m // 2), nn.ReLU(True)]
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_ch, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN — judges local realism rather than the whole frame."""

    def __init__(self, in_ch: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, ndf, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layers + 1):
            prev, mult = mult, min(2 ** n, 8)
            stride = 2 if n < n_layers else 1
            layers += [nn.Conv2d(ndf * prev, ndf * mult, 4, stride=stride, padding=1),
                       nn.InstanceNorm2d(ndf * mult), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(ndf * mult, 1, 4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


@dataclass(frozen=True)
class CycleGANConfig:
    epochs: int = 100
    lr: float = 2e-4
    beta1: float = 0.5
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
    save_every: int = 5
    out_dir: str = "neural/checkpoints/cyclegan"


class CycleGANTrainer:
    """Trainer for unpaired visible<->thermal translation."""

    def __init__(self, cfg: CycleGANConfig | None = None):
        self.cfg = cfg or CycleGANConfig()
        self.dev = get_device()
        self.G = ResnetGenerator().to(self.dev)   # RGB -> IR
        self.F = ResnetGenerator().to(self.dev)   # IR  -> RGB
        self.D_ir = PatchDiscriminator().to(self.dev)
        self.D_rgb = PatchDiscriminator().to(self.dev)
        self.gan_loss = nn.MSELoss()              # LSGAN, more stable than BCE
        self.l1 = nn.L1Loss()
        self.opt_g = torch.optim.Adam(
            itertools.chain(self.G.parameters(), self.F.parameters()),
            lr=self.cfg.lr, betas=(self.cfg.beta1, 0.999))
        self.opt_d = torch.optim.Adam(
            itertools.chain(self.D_ir.parameters(), self.D_rgb.parameters()),
            lr=self.cfg.lr, betas=(self.cfg.beta1, 0.999))

    def _adv(self, pred, is_real: bool):
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.gan_loss(pred, target)

    def step(self, rgb: torch.Tensor, ir: torch.Tensor) -> dict[str, float]:
        rgb, ir = rgb.to(self.dev), ir.to(self.dev)
        c = self.cfg

        # --- generators ---
        self.opt_g.zero_grad()
        fake_ir, fake_rgb = self.G(rgb), self.F(ir)
        loss_g = self._adv(self.D_ir(fake_ir), True) + self._adv(self.D_rgb(fake_rgb), True)
        loss_cycle = self.l1(self.F(fake_ir), rgb) + self.l1(self.G(fake_rgb), ir)
        loss_id = self.l1(self.G(ir), ir) + self.l1(self.F(rgb), rgb)
        total_g = loss_g + c.lambda_cycle * loss_cycle + c.lambda_identity * loss_id
        total_g.backward()
        self.opt_g.step()

        # --- discriminators ---
        self.opt_d.zero_grad()
        loss_d = (
            self._adv(self.D_ir(ir), True) + self._adv(self.D_ir(fake_ir.detach()), False)
            + self._adv(self.D_rgb(rgb), True) + self._adv(self.D_rgb(fake_rgb.detach()), False)
        ) * 0.5
        loss_d.backward()
        self.opt_d.step()

        return {"g": float(total_g), "d": float(loss_d), "cycle": float(loss_cycle)}

    def save(self, tag: str) -> Path:
        out = Path(self.cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"cyclegan_{tag}.pt"
        torch.save({"G": self.G.state_dict(), "F": self.F.state_dict()}, path)
        return path

    def train(self, loader, epochs: int | None = None) -> None:
        epochs = epochs or self.cfg.epochs
        for epoch in range(epochs):
            for i, (rgb, ir) in enumerate(loader):
                stats = self.step(rgb, ir)
                if i % 50 == 0:
                    print(f"epoch {epoch} it {i}  G={stats['g']:.3f} D={stats['d']:.3f} cyc={stats['cycle']:.3f}")
            if (epoch + 1) % self.cfg.save_every == 0:
                print(f"saved {self.save(f'e{epoch + 1}')}")


@torch.no_grad()
def translate(generator: nn.Module, rgb_linear, device: str | None = None):
    """Apply a trained RGB->IR generator to a linear-light image."""
    import numpy as np

    from heron.color import srgb

    dev = device or get_device()
    x = np.clip(srgb.linear_to_srgb(np.asarray(rgb_linear, np.float32)), 0, 1)
    t = torch.from_numpy(x).permute(2, 0, 1)[None].to(dev) * 2 - 1
    y = generator.to(dev).eval()(t)
    out = ((y[0].permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1)
    return srgb.srgb_to_linear(out)
