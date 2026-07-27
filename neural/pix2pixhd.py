"""Pix2PixHD — paired high-resolution RGB->thermal translation (CLAUDE.md §2.2B).

Wang et al. 2018. A coarse-to-fine generator (global net + local enhancer),
multi-scale discriminators, and a feature-matching loss that stabilizes training
at high resolution. Requires *pixel-aligned* RGB/thermal pairs — the hardest data
to obtain, but it gives the sharpest results when available (KAIST Multispectral
and FLIR ADAS both ship aligned pairs; see ``neural/datasets.py``).

Like ``cyclegan``, this is a training module: architecture and loop are complete,
but usable weights require a dataset and substantial compute.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from heron.core.device import get_device
from neural.cyclegan import ResnetBlock


class GlobalGenerator(nn.Module):
    """The coarse global network (operates at half the target resolution)."""

    def __init__(
        self, in_ch: int = 3, out_ch: int = 3, ngf: int = 64, n_down: int = 4,
        n_blocks: int = 9, feature_out: bool = False,
    ):
        """``feature_out`` stops before the RGB head so the LocalEnhancer can add
        the global net's *features* (Pix2PixHD's actual design) rather than its
        image — adding an RGB tensor to a feature map is a channel mismatch."""
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_ch, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        for i in range(n_down):
            m = 2 ** i
            layers += [nn.Conv2d(ngf * m, ngf * m * 2, 3, stride=2, padding=1),
                       nn.InstanceNorm2d(ngf * m * 2), nn.ReLU(True)]
        mult = 2 ** n_down
        layers += [ResnetBlock(ngf * mult) for _ in range(n_blocks)]
        for i in range(n_down):
            m = 2 ** (n_down - i)
            layers += [nn.ConvTranspose2d(ngf * m, ngf * m // 2, 3, stride=2, padding=1, output_padding=1),
                       nn.InstanceNorm2d(ngf * m // 2), nn.ReLU(True)]
        self.out_channels = ngf
        if not feature_out:
            layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_ch, 7), nn.Tanh()]
            self.out_channels = out_ch
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class LocalEnhancer(nn.Module):
    """Wraps the global generator with a full-resolution refinement branch."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3, ngf: int = 32, n_blocks: int = 3):
        super().__init__()
        # global net emits FEATURES (ngf*2 channels) at half resolution
        self.global_net = GlobalGenerator(in_ch, out_ch, ngf * 2, feature_out=True)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.head = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(in_ch, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1), nn.InstanceNorm2d(ngf * 2), nn.ReLU(True),
        )
        self.body = nn.Sequential(*[ResnetBlock(ngf * 2) for _ in range(n_blocks)])
        self.tail = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_ch, 7), nn.Tanh(),
        )

    def forward(self, x):
        coarse = self.global_net(self.downsample(x))     # (B, ngf*2, H/2, W/2)
        local = self.head(x)                             # (B, ngf*2, H/2, W/2)
        coarse = F.interpolate(coarse, size=local.shape[-2:], mode="bilinear", align_corners=False)
        return self.tail(self.body(local + coarse))


class MultiScaleDiscriminator(nn.Module):
    """Several PatchGANs at different scales; returns per-scale feature lists."""

    def __init__(self, in_ch: int = 6, ndf: int = 64, n_scales: int = 3, n_layers: int = 3):
        super().__init__()
        self.n_scales = n_scales
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.nets = nn.ModuleList()
        for _ in range(n_scales):
            seq, mult = nn.ModuleList(), 1
            seq.append(nn.Sequential(nn.Conv2d(in_ch, ndf, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True)))
            for n in range(1, n_layers + 1):
                prev, mult = mult, min(2 ** n, 8)
                stride = 2 if n < n_layers else 1
                seq.append(nn.Sequential(
                    nn.Conv2d(ndf * prev, ndf * mult, 4, stride=stride, padding=1),
                    nn.InstanceNorm2d(ndf * mult), nn.LeakyReLU(0.2, True)))
            seq.append(nn.Sequential(nn.Conv2d(ndf * mult, 1, 4, stride=1, padding=1)))
            self.nets.append(seq)

    def forward(self, x):
        results = []
        for i, net in enumerate(self.nets):
            feats, cur = [], x
            for layer in net:
                cur = layer(cur)
                feats.append(cur)
            results.append(feats)
            if i < self.n_scales - 1:
                x = self.downsample(x)
        return results


@dataclass(frozen=True)
class Pix2PixHDConfig:
    epochs: int = 100
    lr: float = 2e-4
    beta1: float = 0.5
    lambda_feat: float = 10.0
    lambda_l1: float = 10.0
    save_every: int = 5
    out_dir: str = "neural/checkpoints/pix2pixhd"


class Pix2PixHDTrainer:
    def __init__(self, cfg: Pix2PixHDConfig | None = None):
        self.cfg = cfg or Pix2PixHDConfig()
        self.dev = get_device()
        self.G = LocalEnhancer().to(self.dev)
        self.D = MultiScaleDiscriminator().to(self.dev)
        self.gan = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=cfg.lr if cfg else 2e-4, betas=(0.5, 0.999))
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=cfg.lr if cfg else 2e-4, betas=(0.5, 0.999))

    def _adv(self, preds, is_real: bool):
        loss = 0.0
        for feats in preds:
            out = feats[-1]
            target = torch.ones_like(out) if is_real else torch.zeros_like(out)
            loss = loss + self.gan(out, target)
        return loss

    def step(self, rgb: torch.Tensor, ir: torch.Tensor) -> dict[str, float]:
        rgb, ir = rgb.to(self.dev), ir.to(self.dev)
        c = self.cfg
        fake = self.G(rgb)

        # --- generator: adversarial + feature matching + L1 ---
        self.opt_g.zero_grad()
        pred_fake = self.D(torch.cat([rgb, fake], 1))
        with torch.no_grad():
            pred_real = self.D(torch.cat([rgb, ir], 1))
        loss_fm = 0.0
        for pf, pr in zip(pred_fake, pred_real):
            for a, b in zip(pf[:-1], pr[:-1]):
                loss_fm = loss_fm + self.l1(a, b.detach())
        total_g = self._adv(pred_fake, True) + c.lambda_feat * loss_fm + c.lambda_l1 * self.l1(fake, ir)
        total_g.backward()
        self.opt_g.step()

        # --- discriminator ---
        self.opt_d.zero_grad()
        loss_d = 0.5 * (self._adv(self.D(torch.cat([rgb, ir], 1)), True)
                        + self._adv(self.D(torch.cat([rgb, fake.detach()], 1)), False))
        loss_d.backward()
        self.opt_d.step()
        return {"g": float(total_g), "d": float(loss_d)}

    def save(self, tag: str) -> Path:
        out = Path(self.cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"pix2pixhd_{tag}.pt"
        torch.save({"G": self.G.state_dict()}, path)
        return path

    def train(self, loader, epochs: int | None = None) -> None:
        for epoch in range(epochs or self.cfg.epochs):
            for i, (rgb, ir) in enumerate(loader):
                s = self.step(rgb, ir)
                if i % 50 == 0:
                    print(f"epoch {epoch} it {i}  G={s['g']:.3f} D={s['d']:.3f}")
            if (epoch + 1) % self.cfg.save_every == 0:
                print(f"saved {self.save(f'e{epoch + 1}')}")
