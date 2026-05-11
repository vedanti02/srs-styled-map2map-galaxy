"""Differentiable, batched power spectrum P(k) for use as a training loss.

Operates on a batch of displacement fields ``(B, 3, N, N, N)`` and produces
log-binned P(k) estimates ``(B, n_bins)``. Implemented in pure PyTorch so
gradients flow back through the FFT / divergence / binning.

Density estimator: linear ``δ ≈ -∇·Ψ`` via centered finite differences with
periodic wrap. Cheap and exactly matches the offline ``analysis.power_spectrum``
"div" estimator when both run on the same input.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch


def _periodic_div(disp: torch.Tensor, lbox: float) -> torch.Tensor:
    """Compute -divergence of disp via centered diffs with periodic wrap."""
    N = disp.shape[-1]
    cell = lbox / N
    dpx = (torch.roll(disp[:, 0], -1, dims=-3) - torch.roll(disp[:, 0], 1, dims=-3)) / (2.0 * cell)
    dpy = (torch.roll(disp[:, 1], -1, dims=-2) - torch.roll(disp[:, 1], 1, dims=-2)) / (2.0 * cell)
    dpz = (torch.roll(disp[:, 2], -1, dims=-1) - torch.roll(disp[:, 2], 1, dims=-1)) / (2.0 * cell)
    return -(dpx + dpy + dpz)


def _build_k_grid(N: int, lbox: float, device, dtype):
    kx = torch.fft.fftfreq(N, d=lbox / N, device=device, dtype=dtype) * (2 * math.pi)
    return torch.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kx[None, None, :] ** 2)


def _build_bin_assignment(N: int, lbox: float, n_bins: int, device, dtype):
    """One-time setup: returns ``bin_idx`` (N,N,N int) and ``n_modes`` (n_bins,)."""
    kgrid = _build_k_grid(N, lbox, device, dtype)
    k_nyq = math.pi * N / lbox
    k_min = 2 * math.pi / lbox
    edges = torch.logspace(
        math.log10(k_min * 1.01), math.log10(k_nyq), n_bins + 1,
        device=device, dtype=dtype,
    )
    bin_idx = torch.bucketize(kgrid, edges) - 1                # (N,N,N) ints in [-1, n_bins-1]
    bin_idx = bin_idx.clamp(0, n_bins - 1)
    valid = (kgrid >= edges[0]) & (kgrid < edges[-1])
    n_modes = torch.zeros(n_bins, device=device, dtype=dtype)
    n_modes.scatter_add_(0, bin_idx[valid].view(-1), torch.ones_like(bin_idx[valid].view(-1), dtype=dtype))
    return bin_idx, valid, n_modes, edges


class TorchPk:
    """Pre-cached spherical-shell binning. Construct once per (N, lbox, n_bins)."""

    def __init__(self, N: int, lbox: float = 1000.0, n_bins: int = 32,
                 device="cuda", dtype=torch.float32):
        self.N = N
        self.lbox = lbox
        self.n_bins = n_bins
        self.dtype = dtype
        bin_idx, valid, n_modes, edges = _build_bin_assignment(N, lbox, n_bins, device, dtype)
        self.bin_idx = bin_idx
        self.valid = valid
        self.n_modes = n_modes  # (n_bins,)
        self.edges = edges

    @torch.amp.autocast("cuda", enabled=False)
    def __call__(self, disp_or_field: torch.Tensor, is_density: bool = False) -> torch.Tensor:
        """If ``is_density`` is False, ``disp_or_field`` is ``(B,3,N,N,N)`` and the
        density is computed via ``-∇·Ψ``. Else it's already a density ``(B,N,N,N)``.
        Returns log10 P(k) of shape ``(B, n_bins)``.
        """
        # Force fp32 for FFT stability
        x = disp_or_field.float()
        if not is_density:
            delta = _periodic_div(x, self.lbox)
        else:
            delta = x
        if delta.dim() == 5:  # (B,1,N,N,N) edge case
            delta = delta.squeeze(1)
        delta_k = torch.fft.fftn(delta, dim=(-3, -2, -1)) / (self.N ** 3)
        pk_grid = (delta_k.abs() ** 2) * (self.lbox ** 3)               # (B,N,N,N) real

        B = pk_grid.shape[0]
        flat_pk = pk_grid.reshape(B, -1)                                # (B, N^3)
        flat_idx = self.bin_idx.reshape(-1)                             # (N^3,)
        valid_mask = self.valid.reshape(-1)                             # (N^3,)

        idx_b = flat_idx.unsqueeze(0).expand(B, -1)                     # (B, N^3)
        sums = torch.zeros(B, self.n_bins, device=x.device, dtype=torch.float32)
        weights = (valid_mask.float()).unsqueeze(0).expand(B, -1)
        sums.scatter_add_(1, idx_b, flat_pk * weights)
        n = self.n_modes.clamp_min(1.0).unsqueeze(0).expand(B, -1)
        pk_binned = sums / n                                            # (B, n_bins)
        return torch.log10(pk_binned.clamp_min(1e-30))


__all__ = ["TorchPk"]
