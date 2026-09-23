"""Linear-path conditional flow matching between a source built from the LF field
(or pure noise) and the paired HF field.

    x_t = (1 - t) x0 + t x1,     target velocity v* = x1 - x0,
    loss = E || v_theta(x_t, t, cond) - v* ||^2

Sources (the main ablation axis):
  gaussian    : x0 ~ N(0, I)                 standard conditional FM from noise
  lf_white    : x0 = y_lf + s * sigma_z eps   Fotiadis et al. 2024 with the identity encoder;
                                              sigma_z = RMSE(y_lf - y_hf) in model space
  lf_spectral : x0 = y_lf + s * C eps         C colours white noise to the residual power
                                              spectrum of (y_hf - y_lf); large scales are
                                              left at their LF values
`s` is --sigma-scale (1.0 = paper value). With s = 0 the flow from LF is a deterministic
map and can only return ONE field per LF, i.e. the GAN's failure mode by another route.

Samplers integrate dx/dt = v_theta from t=0 to t=1 (Euler, or Heun = 2nd order).
"""
import torch

SOURCES = ("gaussian", "lf_white", "lf_spectral")


def shell_index(n: int, device=None):
    """Integer |k| shell index for an n^3 FFT grid (k in units of the fundamental)."""
    f = torch.fft.fftfreq(n, d=1.0 / n, device=device)  # -n/2 .. n/2-1 integers
    kmag = torch.sqrt(f[:, None, None] ** 2 + f[None, :, None] ** 2 + f[None, None, :] ** 2)
    return torch.round(kmag).long()


def residual_spectrum(residuals: torch.Tensor):
    """Shell-averaged E|FFT(r)|^2 of residual patches r (B,1,n,n,n) -> (n_shells,) table.
    Convention: torch.fft.fftn unnormalised. Colouring white noise eps with
    amp(k) = sqrt(P(k)/n^3) via ifftn(fftn(eps)*amp) reproduces var(r) per voxel."""
    n = residuals.shape[-1]
    fk = torch.fft.fftn(residuals.float(), dim=(-3, -2, -1))
    p = (fk.real ** 2 + fk.imag ** 2).mean(dim=(0, 1))       # (n,n,n)
    idx = shell_index(n, p.device)
    n_sh = int(idx.max().item()) + 1
    s = torch.zeros(n_sh, device=p.device).index_add_(0, idx.flatten(), p.flatten())
    c = torch.zeros(n_sh, device=p.device).index_add_(0, idx.flatten(), torch.ones_like(p.flatten()))
    return s / c.clamp_min(1.0)


def amp_grid_from_table(table: torch.Tensor, n: int):
    idx = shell_index(n, table.device).clamp_max(len(table) - 1)
    return torch.sqrt(table[idx] / float(n ** 3))


class Interpolant:
    def __init__(self, source: str, sigma_z: float = 1.0, sigma_scale: float = 1.0,
                 spec_table=None, grid: int = 64):
        assert source in SOURCES, source
        self.source = source
        self.sigma_z = float(sigma_z)
        self.sigma_scale = float(sigma_scale)
        self.grid = int(grid)
        self.spec_table = None if spec_table is None else torch.as_tensor(spec_table).float()
        self._amp = None
        if source == "lf_spectral":
            assert self.spec_table is not None, "lf_spectral needs a residual spectrum table"

    def state(self):
        return {"source": self.source, "sigma_z": self.sigma_z, "sigma_scale": self.sigma_scale,
                "spec_table": None if self.spec_table is None else self.spec_table.cpu(),
                "grid": self.grid}

    @classmethod
    def from_state(cls, s):
        return cls(**s)

    def _amp_for(self, n, device):
        if self._amp is None or self._amp.shape[-1] != n or self._amp.device != device:
            self._amp = amp_grid_from_table(self.spec_table.to(device), n)
        return self._amp

    def colour(self, eps: torch.Tensor) -> torch.Tensor:
        amp = self._amp_for(eps.shape[-1], eps.device)
        fk = torch.fft.fftn(eps.float(), dim=(-3, -2, -1)) * amp
        return torch.fft.ifftn(fk, dim=(-3, -2, -1)).real.to(eps.dtype)

    def sample_source(self, y_lf: torch.Tensor, generator=None) -> torch.Tensor:
        eps = torch.randn(y_lf.shape, device=y_lf.device, dtype=y_lf.dtype, generator=generator)
        if self.source == "gaussian":
            return self.sigma_scale * eps      # sigma_scale = sampling "temperature" (1.0 = as trained)
        if self.source == "lf_white":
            return y_lf + self.sigma_scale * self.sigma_z * eps
        return y_lf + self.sigma_scale * self.colour(eps)

    @staticmethod
    def path(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        tt = t.view(-1, *([1] * (x0.dim() - 1))).to(x0.dtype)
        return (1.0 - tt) * x0 + tt * x1, x1 - x0

    def __repr__(self):
        return f"Interpolant({self.source}, sigma_z={self.sigma_z:.4f}, scale={self.sigma_scale})"


def sample_t(batch: int, dist: str, device, generator=None, shift: float = 0.0) -> torch.Tensor:
    if dist == "uniform":
        return torch.rand(batch, device=device, generator=generator)
    if dist == "logitnormal":   # SD3-style; shift>0 moves mass toward t=1 where fine structure forms
        return torch.sigmoid(torch.randn(batch, device=device, generator=generator) + shift)
    raise ValueError(dist)


@torch.no_grad()
def integrate(v_fn, x0: torch.Tensor, n_steps: int, method: str = "heun", sde_gamma: float = 1.0,
              generator=None):
    """Integrate from t=0 to t=1 on a uniform grid. v_fn(x, t) with t a (B,) tensor.
    euler / heun: the probability-flow ODE dx = v dt (Heun = 2nd order, Euler on the final step).
    sde: same marginals via dx = [v + (s_t^2/2) grad log p_t] dt + s_t dW with s_t = gamma (1-t).
         VALID ONLY for the unit-Gaussian source with independent coupling, where
         grad log p_t(x_t) = (t v - x_t)/(1-t)  (from E[x1|x_t] = x_t + (1-t) v)."""
    x = x0
    B = x.shape[0]
    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=x.device, dtype=x.dtype)
    for i in range(n_steps):
        t0, t1 = ts[i], ts[i + 1]
        dt = t1 - t0
        v0 = v_fn(x, t0.expand(B))
        if method == "sde":
            omt = 1.0 - t0
            drift = v0 + 0.5 * sde_gamma ** 2 * omt * (t0 * v0 - x)      # v + s^2/2 * score
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            x = x + dt * drift + sde_gamma * omt * torch.sqrt(dt) * noise
        elif method == "euler" or i == n_steps - 1:
            x = x + dt * v0
        elif method == "heun":
            xe = x + dt * v0
            v1 = v_fn(xe, t1.expand(B))
            x = x + 0.5 * dt * (v0 + v1)
        else:
            raise ValueError(method)
    return x


def project_lowk(x: torch.Tensor, ref: torch.Tensor, kc: int) -> torch.Tensor:
    """Replace Fourier modes with |k| < kc (units of the box fundamental) of x by those of ref.
    Diagnostic constraint: with ref = LF, large scales are guaranteed to be LF's."""
    if kc <= 0:
        return x
    n = x.shape[-1]
    mask = (shell_index(n, x.device) < kc)
    fx = torch.fft.fftn(x.float(), dim=(-3, -2, -1))
    fr = torch.fft.fftn(ref.float(), dim=(-3, -2, -1))
    fx = torch.where(mask, fr, fx)
    return torch.fft.ifftn(fx, dim=(-3, -2, -1)).real.to(x.dtype)
