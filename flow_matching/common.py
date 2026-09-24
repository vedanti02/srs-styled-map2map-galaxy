"""Shared helpers: EMA, checkpoint (de)serialisation, per-box generation, validation metrics."""
import copy
import numpy as np
import torch

from flow_matching.space import ModelSpace
from flow_matching.interpolant import Interpolant, integrate, project_lowk
from flow_matching.unet3d import build_unet
from flow_matching.data import extract_patch, stitch_torch, N_PATCHES, PATCH


class EMA:
    """Exponential moving average of parameters with the usual warm-up on the decay."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.step = 0

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        for ps, pm in zip(self.shadow.parameters(), model.parameters()):
            ps.mul_(d).add_(pm.detach(), alpha=1.0 - d)
        for bs, bm in zip(self.shadow.buffers(), model.buffers()):
            bs.copy_(bm)

    def state_dict(self):
        return {"decay": self.decay, "step": self.step, "model": self.shadow.state_dict()}

    def load_state_dict(self, s):
        self.decay = s["decay"]; self.step = s["step"]; self.shadow.load_state_dict(s["model"])


def build_from_ckpt(ck, device):
    """-> (net with EMA weights, space, interp, args dict)."""
    a = ck["args"]
    space = ModelSpace.from_state(ck["space"])
    interp = Interpolant.from_state(ck["interp"])
    net = build_unet(a, in_ch=(1 + int(a.get("n_cond", 1))) if a["cond"] else 1).to(device)
    sd = ck["ema"]["model"] if ("ema" in ck and ck["ema"] is not None) else ck["model"]
    net.load_state_dict(sd)
    net.eval()
    return net, space, interp, a


def make_v_fn(net, cond, amp_dtype=None):
    def v_fn(x, t):
        if amp_dtype is not None:
            with torch.autocast("cuda", dtype=amp_dtype):
                return net(x, t, cond).float()
        return net(x, t, cond)
    return v_fn


@torch.no_grad()
def generate_box(net, space, interp, lr_counts, device, n_draws=1, steps=32, method="heun",
                 seed=0, cond=True, lowk_kc=0, amp_dtype=None, return_model_space=False, decode_shift=0.0, sde_gamma=1.0,
                 full_box=False, extra_counts=None):
    """lr_counts: (1,128,128,128) float32 physical LF counts (pad=0 protocol).
    full_box: run the (fully convolutional, circularly padded) net on the whole periodic box at once
    instead of 8 independent 64^3 patches, so large scales and patch faces see correct context.
    extra_counts: (1,128,128,128) output of a deterministic corrector; required by the gan_white source
    (it is both the flow's starting point and a second conditioning channel).
    -> list of n_draws (1,128,128,128) float32 count cubes (integers), plus optional model-space cubes."""
    if full_box:
        lr_p = torch.from_numpy(np.asarray(lr_counts, np.float32)[None]).to(device)   # (1,1,128,128,128)
    else:
        lr_p = torch.from_numpy(np.stack([extract_patch(lr_counts, p, 0) for p in range(N_PATCHES)])).to(device)
    y_lf = space.forward(lr_p, dequant=False)                      # exact transform for conditioning/source
    y_base = None
    if interp.source == "gan_white":
        assert extra_counts is not None, "gan_white needs extra_counts (the GAN output)"
        ex = np.asarray(extra_counts, np.float32)
        ex_p = ex[None] if full_box else np.stack([extract_patch(ex, p, 0) for p in range(N_PATCHES)])
        y_base = space.forward(torch.from_numpy(ex_p).to(device), dequant=False)
    cond_t = (torch.cat([y_lf, y_base], dim=1) if y_base is not None else y_lf) if cond else None
    v_fn = make_v_fn(net, cond_t, amp_dtype)
    outs, outs_model = [], []
    for d in range(n_draws):
        g = torch.Generator(device=device).manual_seed(int(seed) * 1_000_003 + d)
        x0 = interp.sample_source(y_lf, generator=g, y_base=y_base)
        x1 = integrate(v_fn, x0, steps, method, sde_gamma=sde_gamma, generator=g)  # (8,1,64,64,64) model space
        box_m = x1[0] if full_box else stitch_torch(x1)             # (1,128^3)
        if lowk_kc > 0:
            box_m = project_lowk(box_m, y_lf[0] if full_box else stitch_torch(y_lf), lowk_kc)
        outs.append(space.inverse(box_m, decode_shift).cpu().numpy())
        if return_model_space:
            outs_model.append(box_m.cpu().numpy())
    return (outs, outs_model) if return_model_space else outs


def crps_ensemble(samples: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Fair CRPS per voxel from K samples. samples (K,*), obs (*). K=1 -> MAE."""
    K = samples.shape[0]
    term1 = (samples - obs[None]).abs().mean(0)
    if K == 1:
        return term1
    diff = (samples[:, None] - samples[None]).abs().sum(dim=(0, 1))
    return term1 - diff / (2.0 * K * (K - 1))


@torch.no_grad()
def box_metrics(sr_list, hr_counts, lr_counts, pk128, device):
    """sr_list: list of (1,128^3) count cubes (draws). hr/lr: (1,128^3) counts.
    Returns dict of scalar metrics in exact log1p space (protocol space) + P(k) rms."""
    from data.patch_dataset_cmass import counts_to_delta
    S = torch.from_numpy(np.stack(sr_list)).to(device)             # (K,1,128^3)
    H = torch.from_numpy(hr_counts).to(device)                     # (1,128^3)
    lS, lH = torch.log1p(S), torch.log1p(H)
    l1_draw = (lS - lH[None]).abs().mean().item()
    l1_mean = (torch.log1p(S.mean(0)) - lH).abs().mean().item()
    crps = crps_ensemble(lS[:, 0], lH[0]).mean().item()
    spread = lS.std(0, unbiased=False).mean().item() if S.shape[0] > 1 else 0.0
    cnt_err = ((S.sum(dim=(1, 2, 3, 4)) - H.sum()) / H.sum()).mean().item()
    # P(k) rms of draw 0 vs HR (held-out check only; never trained on)
    d_s = counts_to_delta(S[:1])                                   # per-box nbar
    d_h = counts_to_delta(H[None])
    lp_s = pk128(d_s, is_density=True); lp_h = pk128(d_h, is_density=True)
    m = (lp_h > -10).float()
    pk_rms = ((((lp_s - lp_h) ** 2) * m).sum() / m.sum().clamp_min(1.0)).sqrt().item()
    return {"l1": l1_draw, "l1_mean": l1_mean, "crps": crps, "spread": spread,
            "cnt_err": cnt_err, "pk_rms": pk_rms}
