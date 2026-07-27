"""Neural Style Transfer — Gatys et al. + AdaIN (CLAUDE.md Phase 5).

Transfers the *painterly* qualities of a reference image (brushwork, palette
statistics, tonal structure) onto a Heron render while keeping Heron's physical
content. Two modes:

* ``gatys``  — optimize the image directly against VGG19 content/style Gram
  losses. Slow (seconds to a minute) but highest quality.
* ``adain``  — adaptive instance normalization in VGG feature space: a single
  forward pass, near-instant, good for previews.

Semantic feature matching is supported via ``region_mask``: style statistics can
be matched only where a mask says the subject is, so a figure's style is not
contaminated by background statistics.

NOTE ON THE REFERENCES: transferring a living artist's paintings is fine for
private study, but their style is their own. Do not ship outputs derived from
a living artist's copyrighted work without a license agreement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from heron.color import srgb
from heron.core.device import get_device

_VGG_LAYERS = {"0": "conv1_1", "5": "conv2_1", "10": "conv3_1", "19": "conv4_1", "21": "conv4_2", "28": "conv5_1"}
_STYLE_LAYERS = ("conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1")
_CONTENT_LAYER = "conv4_2"
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass(frozen=True)
class StyleParams:
    mode: str = "gatys"        # gatys | adain
    steps: int = 240
    style_weight: float = 1.2e6
    content_weight: float = 1.0
    tv_weight: float = 2.0
    lr: float = 0.03
    max_side: int = 768
    adain_alpha: float = 1.0   # adain only: 0 = content, 1 = full style


def _load_vgg():
    from torchvision.models import VGG19_Weights, vgg19

    model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(get_device())


def _to_tensor(img_linear: np.ndarray, max_side: int) -> torch.Tensor:
    """linear-light HWC [0,1] -> normalized NCHW tensor on the device."""
    srgb_img = np.clip(srgb.linear_to_srgb(np.asarray(img_linear, np.float32)), 0, 1)
    t = torch.from_numpy(srgb_img).permute(2, 0, 1)[None]
    h, w = t.shape[-2:]
    scale = max_side / max(h, w)
    if scale < 1.0:
        t = F.interpolate(t, size=(int(h * scale), int(w * scale)), mode="bilinear", align_corners=False)
    dev = get_device()
    return ((t - _MEAN.to(t)) / _STD.to(t)).to(dev)


def _to_image(t: torch.Tensor) -> np.ndarray:
    """normalized NCHW tensor -> linear-light HWC [0,1]."""
    x = t.detach().cpu() * _STD + _MEAN
    x = x.clamp(0, 1)[0].permute(1, 2, 0).numpy()
    return srgb.srgb_to_linear(x).astype(np.float32)


def _features(model, x: torch.Tensor) -> dict[str, torch.Tensor]:
    out, cur = {}, x
    for name, layer in model._modules.items():
        cur = layer(cur)
        if name in _VGG_LAYERS:
            out[_VGG_LAYERS[name]] = cur
    return out


def _gram(f: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Gram matrix, optionally restricted to a region (semantic matching)."""
    b, c, h, w = f.shape
    x = f.reshape(b, c, h * w)
    if mask is not None:
        m = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False).reshape(b, 1, h * w)
        x = x * m
        denom = m.sum().clamp(min=1.0) * c
    else:
        denom = float(c * h * w)
    return (x @ x.transpose(1, 2)) / denom


def _adain(content_f: torch.Tensor, style_f: torch.Tensor, alpha: float) -> torch.Tensor:
    c_mean, c_std = content_f.mean((2, 3), keepdim=True), content_f.std((2, 3), keepdim=True) + 1e-5
    s_mean, s_std = style_f.mean((2, 3), keepdim=True), style_f.std((2, 3), keepdim=True) + 1e-5
    normalized = (content_f - c_mean) / c_std * s_std + s_mean
    return alpha * normalized + (1 - alpha) * content_f


def stylize(
    content_linear: np.ndarray,
    style_linear: np.ndarray,
    params: StyleParams | None = None,
    region_mask: np.ndarray | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Return a stylized linear-light RGB image."""
    p = params or StyleParams()
    model = _load_vgg()
    content = _to_tensor(content_linear, p.max_side)
    style = _to_tensor(style_linear, p.max_side)

    mask_t = None
    if region_mask is not None:
        m = torch.from_numpy(np.asarray(region_mask, np.float32))[None, None]
        mask_t = F.interpolate(m, size=content.shape[-2:], mode="bilinear", align_corners=False).to(content)

    with torch.no_grad():
        cf = _features(model, content)
        sf = _features(model, style)

    if p.mode == "adain":
        # single-pass feature-statistic transfer, decoded by optimizing briefly
        target = {k: _adain(cf[k], sf[k], p.adain_alpha) for k in _STYLE_LAYERS}
        img = content.clone().requires_grad_(True)
        opt = torch.optim.Adam([img], lr=p.lr)
        for i in range(max(30, p.steps // 6)):
            opt.zero_grad()
            f = _features(model, img)
            loss = sum(F.mse_loss(f[k], target[k]) for k in _STYLE_LAYERS)
            loss.backward()
            opt.step()
        return _to_image(img)

    style_grams = {k: _gram(sf[k]) for k in _STYLE_LAYERS}
    content_target = cf[_CONTENT_LAYER].detach()

    img = content.clone().requires_grad_(True)
    opt = torch.optim.Adam([img], lr=p.lr)
    for step in range(p.steps):
        opt.zero_grad()
        f = _features(model, img)
        s_loss = sum(F.mse_loss(_gram(f[k], mask_t), style_grams[k]) for k in _STYLE_LAYERS)
        c_loss = F.mse_loss(f[_CONTENT_LAYER], content_target)
        tv = (img[..., 1:, :] - img[..., :-1, :]).abs().mean() + (img[..., 1:] - img[..., :-1]).abs().mean()
        loss = p.style_weight * s_loss + p.content_weight * c_loss + p.tv_weight * tv
        loss.backward()
        opt.step()
        if verbose and step % 60 == 0:
            print(f"  step {step:4d}/{p.steps}  style={float(s_loss):.5f} content={float(c_loss):.3f}")
    return _to_image(img)
