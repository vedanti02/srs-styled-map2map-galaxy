"""CPU tests on synthetic cubes (no data access needed). Run:
    python -m flow_matching.tests.test_fm        (plain)     or     pytest flow_matching/tests
"""
import math, sys, traceback
import numpy as np
import torch

from flow_matching.space import ModelSpace
from flow_matching.augment import all_ops, apply_op, random_op
from flow_matching.interpolant import (Interpolant, integrate, project_lowk, residual_spectrum,
                                       amp_grid_from_table, shell_index, sample_t)
from flow_matching.unet3d import UNet3D
from flow_matching.common import EMA, crps_ensemble
from flow_matching.data import stitch_torch

torch.manual_seed(0)


def sparse_counts(shape, mean=0.15, gen=None):
    """Poisson-like sparse integer field with clustering (log-normal rate)."""
    g = gen or torch.Generator().manual_seed(1)
    rate = torch.exp(torch.randn(shape, generator=g) * 1.0) * mean
    return torch.poisson(rate, generator=g)


def test_space_roundtrip():
    n = sparse_counts((2, 1, 16, 16, 16)); n[0, 0, 0, 0, 0] = 199
    sp = ModelSpace(0.1, 0.3, dequant=True)
    y = sp.forward(n, generator=torch.Generator().manual_seed(0))
    assert torch.equal(sp.inverse(y), n), "dequantised round-trip must be exact (floor)"
    sp2 = ModelSpace(0.1, 0.3, dequant=False)
    assert torch.equal(sp2.inverse(sp2.forward(n)), n)
    assert torch.allclose(sp.forward(n, dequant=False), (torch.log1p(n) - 0.1) / 0.3)
    # inverse tolerates NaN/inf and clamps to cmax
    bad = torch.tensor([[[[[float("nan"), float("inf"), -50.0, 50.0]]]]])
    out = sp.inverse(bad); assert torch.isfinite(out).all() and out.max() <= sp.cmax and out.min() >= 0


def test_augment_group_sizes_and_pairing():
    assert len(all_ops("none")) == 1 and len(all_ops("los16")) == 16 and len(all_ops("full48")) == 48
    x = torch.randn(3, 1, 8, 8, 8); y = 2 * x + 1
    for op in all_ops("full48"):
        xa, ya = apply_op(x, op), apply_op(y, op)
        assert xa.shape == x.shape and torch.allclose(ya, 2 * xa + 1), "LF/HF voxel pairing must survive"
    for op in all_ops("los16"):
        assert op[0][2] == 2, "los16 must keep the last axis"
    # the 48 ops are distinct
    outs = {apply_op(torch.arange(27.).view(1, 1, 3, 3, 3), op).flatten().tolist().__str__() for op in all_ops("full48")}
    assert len(outs) == 48
    random_op("full48", torch.Generator().manual_seed(0))


def test_interpolant_paths_and_sources():
    y_lf = torch.randn(4, 1, 8, 8, 8); y_hf = torch.randn(4, 1, 8, 8, 8)
    for src in ("gaussian", "lf_white"):
        it = Interpolant(src, sigma_z=0.5)
        x0 = it.sample_source(y_lf, torch.Generator().manual_seed(0))
        xt0, v = it.path(x0, y_hf, torch.zeros(4)); xt1, _ = it.path(x0, y_hf, torch.ones(4))
        assert torch.allclose(xt0, x0) and torch.allclose(xt1, y_hf) and torch.allclose(v, y_hf - x0)
    it = Interpolant("lf_white", sigma_z=0.5)
    big = torch.zeros(64, 1, 8, 8, 8)
    s = it.sample_source(big, torch.Generator().manual_seed(0)).std().item()
    assert abs(s - 0.5) < 0.02, s
    assert Interpolant("gaussian", sigma_z=0.5).sample_source(big, torch.Generator().manual_seed(0)).std().item() - 1 < 0.05
    # state round-trip
    st = it.state(); it2 = Interpolant.from_state(st); assert it2.source == "lf_white" and it2.sigma_z == 0.5
    for d in ("uniform", "logitnormal"):
        t = sample_t(1000, d, "cpu"); assert t.min() >= 0 and t.max() <= 1


def test_spectral_source_matches_residual_spectrum():
    n = 16
    g = torch.Generator().manual_seed(3)
    # residual with power only in shells 3..6 (band-limited), zero at low k
    idx = shell_index(n)
    band = ((idx >= 3) & (idx <= 6)).float()
    fk = torch.fft.fftn(torch.randn(256, 1, n, n, n, generator=g), dim=(-3, -2, -1)) * band
    r = torch.fft.ifftn(fk, dim=(-3, -2, -1)).real
    tab = residual_spectrum(r)
    assert tab[:3].abs().max() < 1e-6 and tab[3:7].min() > 0
    it = Interpolant("lf_spectral", sigma_z=1.0, spec_table=tab, grid=n)
    eps = torch.randn(256, 1, n, n, n, generator=g)
    c = it.colour(eps)
    # per-voxel variance reproduced
    assert abs(c.var().item() / r.var().item() - 1) < 0.1, (c.var().item(), r.var().item())
    # no power at low k
    fc = torch.fft.fftn(c, dim=(-3, -2, -1)); pc = (fc.real ** 2 + fc.imag ** 2).mean(0)[0]
    assert pc[idx < 3].max() < 1e-6 * pc[idx == 4].mean()


def test_unet_shapes_and_periodic_equivariance():
    net = UNet3D(in_ch=2, out_ch=1, base=8, mults=(1, 2, 2, 4), num_res=1, attn_levels=(3,)).eval()
    x = torch.randn(2, 1, 16, 16, 16); c = torch.randn(2, 1, 16, 16, 16); t = torch.rand(2)
    with torch.no_grad():
        y = net(x, t, c)
        assert y.shape == (2, 1, 16, 16, 16)
        # total stride 2^3 = 8: rolling by 8 must commute exactly with the net (circular convs)
        yr = net(torch.roll(x, 8, dims=2), t, torch.roll(c, 8, dims=2))
        assert torch.allclose(torch.roll(y, 8, dims=2), yr, atol=1e-4), (torch.roll(y, 8, dims=2) - yr).abs().max()
    # zero-init output: initial velocity is exactly 0
    net0 = UNet3D(in_ch=1, out_ch=1, base=8, mults=(1, 2), num_res=1, attn_levels=(1,))
    assert net0(x, t).abs().max().item() == 0.0
    n_par = sum(p.numel() for p in UNet3D(in_ch=2, base=32, mults=(1, 2, 4, 8), num_res=2, attn_levels=(3,)).parameters())
    print(f"  default UNet params: {n_par/1e6:.1f}M")


def test_sampler_on_analytic_gaussian_flow():
    """Independent coupling x0~N(0,1), x1~N(mu,s^2): the marginal velocity E[x1-x0|x_t] is linear
    in x_t and analytic. Integrating it must map N(0,1) samples to N(mu, s^2)."""
    mu, s = 1.5, 0.5
    def v_fn(x, t):
        tt = t.view(-1, 1)
        var_t = (1 - tt) ** 2 + tt ** 2 * s ** 2          # var(x_t)
        cov = -(1 - tt) + tt * s ** 2                      # cov(x1-x0, x_t)
        return mu + cov / var_t * (x - tt * mu)
    x0 = torch.randn(200000, 1)
    for method, steps, tol in (("heun", 64, 0.02), ("euler", 400, 0.03)):
        x1 = integrate(v_fn, x0, steps, method)
        assert abs(x1.mean().item() - mu) < tol and abs(x1.std().item() - s) < tol, (method, x1.mean(), x1.std())


def test_project_lowk():
    n = 16
    x = torch.randn(1, 1, n, n, n); ref = torch.randn(1, 1, n, n, n)
    out = project_lowk(x, ref, kc=3)
    idx = shell_index(n)
    fo, fx, fr = (torch.fft.fftn(a, dim=(-3, -2, -1))[0, 0] for a in (out, x, ref))
    assert torch.allclose(fo[idx < 3], fr[idx < 3], atol=1e-4) and torch.allclose(fo[idx >= 3], fx[idx >= 3], atol=1e-4)
    assert torch.equal(project_lowk(x, ref, 0), x)


def test_stitch_torch_matches_numpy():
    from flow_matching.data import stitch_patches, extract_patch
    box = np.random.rand(1, 128, 128, 128).astype(np.float32)
    patches = np.stack([extract_patch(box, p, 0) for p in range(8)])
    a = stitch_patches(patches); b = stitch_torch(torch.from_numpy(patches)).numpy()
    assert np.array_equal(a, box) and np.array_equal(b, box)


def test_ema_and_crps():
    net = torch.nn.Linear(3, 1); ema = EMA(net, 0.9)
    with torch.no_grad(): net.weight.fill_(1.0)
    ema.update(net); ema.update(net)
    assert ema.shadow.weight.mean().item() > 0.0 and ema.shadow.weight.mean().item() < 1.0
    obs = torch.zeros(1000); s = torch.randn(8, 1000)
    c = crps_ensemble(s, obs).mean().item()
    # fair CRPS of N(0,1) ensemble vs a point at 0: E|X| - E|X-X'|/2 = sqrt(2/pi) - 1/sqrt(pi) ~ 0.2337
    assert abs(c - (math.sqrt(2 / math.pi) - 1 / math.sqrt(math.pi))) < 0.03, c
    assert torch.allclose(crps_ensemble(s[:1], obs), (s[0] - obs).abs())


def test_train_step_end_to_end_tiny():
    """One optimisation step on random sparse counts through the whole train path (16^3 cubes)."""
    from flow_matching.augment import random_op, apply_op
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(0)
    lr = sparse_counts((8, 1, 16, 16, 16), gen=g); hr = sparse_counts((8, 1, 16, 16, 16), gen=g)
    sp = ModelSpace(0.1, 0.3, True)
    r = sp.forward(hr) - sp.forward(lr, dequant=False)
    tab = residual_spectrum(r)
    for src in ("gaussian", "lf_white", "lf_spectral"):
        it = Interpolant(src, sigma_z=r.pow(2).mean().sqrt().item(), spec_table=tab, grid=16)
        net = UNet3D(in_ch=2, base=8, mults=(1, 2, 2, 4), num_res=1, attn_levels=(3,))
        opt = torch.optim.AdamW(net.parameters(), 1e-3); ema = EMA(net)
        op = random_op("full48", g); a, b = apply_op(lr, op), apply_op(hr, op)
        y_hf = sp.forward(a * 0 + b); y_lf = sp.forward(a, dequant=False)
        x0 = it.sample_source(y_lf); t = sample_t(8, "uniform", "cpu")
        x_t, v_star = it.path(x0, y_hf, t)
        loss = F.mse_loss(net(x_t, t, y_lf), v_star); loss.backward(); opt.step(); ema.update(net)
        assert torch.isfinite(loss)
        with torch.no_grad():
            x1 = integrate(lambda x, tt: ema.shadow(x, tt, y_lf), x0, 4, "heun")
            cnt = sp.inverse(x1)
        assert cnt.shape == hr.shape and (cnt >= 0).all() and torch.equal(cnt, cnt.floor())




def test_sde_sampler_matches_marginals():
    """Same analytic Gaussian flow as the ODE test: the SDE with score (t v - x)/(1-t) must reach N(mu, s^2)."""
    mu, s = 1.5, 0.5
    def v_fn(x, t):
        tt = t.view(-1, 1); var_t = (1 - tt) ** 2 + tt ** 2 * s ** 2; cov = -(1 - tt) + tt * s ** 2
        return mu + cov / var_t * (x - tt * mu)
    x0 = torch.randn(200000, 1)
    for gm in (0.5, 1.5):
        x1 = integrate(v_fn, x0, 200, "sde", sde_gamma=gm, generator=torch.Generator().manual_seed(0))
        assert abs(x1.mean().item() - mu) < 0.03 and abs(x1.std().item() - s) < 0.03, (gm, x1.mean(), x1.std())


def test_loss_weights_and_tshift():
    import sys
    sys.path.insert(0, ".")
    from flow_matching.train_fm import loss_weights
    n = sparse_counts((4, 1, 8, 8, 8))
    w = loss_weights(n, 2.0)
    assert w.shape == n.shape and abs(w.mean().item() - 1.0) < 1e-5 and w.min() > 0
    assert (w[n > 0].mean() > w[n == 0].mean()), "dense voxels must get more weight"
    t_late = sample_t(20000, "logitnormal", "cpu", shift=1.5); t_mid = sample_t(20000, "logitnormal", "cpu")
    assert t_late.mean() > t_mid.mean() + 0.15


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    fails = 0
    for fn in TESTS:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception:
            fails += 1; print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(TESTS)-fails}/{len(TESTS)} passed")
    sys.exit(1 if fails else 0)
