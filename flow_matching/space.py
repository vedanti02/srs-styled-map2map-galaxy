"""Model space for integer count fields.

    y = (log1p(n + u) - mean) / std,   u ~ U[0,1) if dequantised else 0

Dequantisation turns the lattice-supported count distribution into a density the
flow can represent; the inverse is exact:  n = floor(expm1(y*std + mean)).
Without dequantisation the inverse is round(). The LF conditioning channel is
always the exact (non-dequantised) transform of the LF counts.
"""
import torch


class ModelSpace:
    def __init__(self, mean: float, std: float, dequant: bool = True, cmax: float = 200.0):
        self.mean = float(mean)
        self.std = float(std)
        self.dequant = bool(dequant)
        self.cmax = float(cmax)

    def state(self):
        return {"mean": self.mean, "std": self.std, "dequant": self.dequant, "cmax": self.cmax}

    @classmethod
    def from_state(cls, s):
        return cls(**s)

    def forward(self, counts: torch.Tensor, dequant=None, generator=None) -> torch.Tensor:
        """Physical counts (float tensor of integers) -> model space."""
        dq = self.dequant if dequant is None else bool(dequant)
        n = counts
        if dq:
            u = torch.rand(n.shape, device=n.device, dtype=n.dtype, generator=generator)
            n = n + u
        return (torch.log1p(n) - self.mean) / self.std

    def inverse(self, y: torch.Tensor, shift: float = 0.0) -> torch.Tensor:
        """Model space -> physical counts (float tensor of integers, clamped to [0, cmax]).
        `shift` (dequantised models only) raises the floor threshold: n = floor(z - shift).
        Diagnosed 2026-09-13: with ~87% empty voxels, model smear across the floor boundary
        creates spurious halos (+9% count, T(k)~0.90 at all k); shift~0.07 cancels it.
        Prefer retraining with dequant=False (round decoder is symmetric under model error)."""
        z = torch.expm1(y * self.std + self.mean)
        z = torch.nan_to_num(z, nan=0.0, posinf=self.cmax, neginf=0.0)
        # +1e-4 guards floor() against expm1(log1p(n+u)) landing at n+u-eps in float32
        n = torch.floor(z - shift + 1e-4) if self.dequant else torch.round(z)
        return n.clamp(0.0, self.cmax)

    def __repr__(self):
        return f"ModelSpace(mean={self.mean:.4f}, std={self.std:.4f}, dequant={self.dequant}, cmax={self.cmax})"
