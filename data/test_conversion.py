#!/usr/bin/env python3
"""
test_conversion.py
===================
Unit tests for conversion.py.  Every expected value here is computed by hand in
the comments next to the assertion, so the tests double as worked examples.

Run with:
    python -m unittest data/test_conversion.py -v
(from the repo's `data/` directory, or with `data/` on PYTHONPATH)
"""

import json
import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np

import conversion as c
import density_histogram as dh

logging.getLogger("conversion").addHandler(logging.NullHandler())


class TestBuildGlobalVoxelGrid(unittest.TestCase):
    def test_small_grid_matches_docstring_example(self):
        # N_global=3, voxel_side=10, box_length=30 -> coords {0,10,20} per axis.
        vc = c.build_global_voxel_grid(3, 10.0, 30.0)
        self.assertEqual(vc.shape, (3, 3, 3, 3))
        # voxel (0,1,2) -> [0., 10., 20.] (exactly as stated in the docstring)
        np.testing.assert_allclose(vc[0, 1, 2], [0.0, 10.0, 20.0])
        # voxel (2,2,2) -> [20., 20., 20.]
        np.testing.assert_allclose(vc[2, 2, 2], [20.0, 20.0, 20.0])

    def test_wraps_at_box_length(self):
        # N_global=4, voxel_side=10, box_length=30: raw coord for gx=3 is 30,
        # which wraps to 0 mod 30.
        vc = c.build_global_voxel_grid(4, 10.0, 30.0)
        np.testing.assert_allclose(vc[3, 0, 0], [0.0, 0.0, 0.0])


class TestPeriodicDistance(unittest.TestCase):
    def test_disp_vec_takes_shortest_path_around_box(self):
        # a=1, b=99, L=100: raw diff = 1-99 = -98.
        # round(-98/100) = round(-0.98) = -1, so wrapped = -98 - 100*(-1) = 2.
        # i.e. going 99 -> 100/0 -> 1 is distance 2, shorter than the raw 98.
        a = np.array([1.0])
        b = np.array([99.0])
        d = c.periodic_disp_vec(a, b, 100.0)
        np.testing.assert_allclose(d, [2.0])

    def test_dist_sq_matches_disp_vec_squared(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([99.0, 0.0, 0.0])
        # squared distance = 2**2 + 0 + 0 = 4
        self.assertAlmostEqual(float(c.periodic_dist_sq(a, b, 100.0)), 4.0)


class TestPatchCenterAndMeshConsistency(unittest.TestCase):
    def test_even_patch_size_center_is_between_two_voxels(self):
        # stride=4, voxel_side=10, i=j=k=2, Ni=Nj=Nk=4 (even).
        # Patch occupies global voxel indices 8,9,10,11 -> physical 80,90,100,110.
        # Geometric centre of that span = mean(80,90,100,110) = 95.
        center = c.generate_patch_center(2, 2, 2, 4, 4, 4, stride=4,
                                          voxel_side=10.0, box_length=1000.0)
        np.testing.assert_allclose(center, [95.0, 95.0, 95.0])

        # build_voxel_mesh around that centre must reproduce the true voxel
        # centres 80, 90, 100, 110 along each axis exactly.
        mesh = c.build_voxel_mesh(center, 10.0, 4, 4, 4, 1000.0)
        np.testing.assert_allclose(mesh[:, 0, 0, 0], [80.0, 90.0, 100.0, 110.0])
        np.testing.assert_allclose(mesh[0, :, 0, 1], [80.0, 90.0, 100.0, 110.0])

    def test_odd_patch_size_center_lands_on_a_voxel(self):
        # stride=5, voxel_side=2, i=j=k=1, Ni=Nj=Nk=5 (odd).
        # Patch occupies global indices 5,6,7,8,9 -> physical 10,12,14,16,18.
        # Centre index = 1*5 + (5-1)/2 = 7 -> physical 14 (exactly voxel 7's centre).
        center = c.generate_patch_center(1, 1, 1, 5, 5, 5, stride=5,
                                          voxel_side=2.0, box_length=1000.0)
        np.testing.assert_allclose(center, [14.0, 14.0, 14.0])

    def test_clipped_last_patch_uses_actual_size_not_max_N(self):
        # This is the bug-1 regression case: a last patch with actual size 3
        # (smaller than the max patch size, e.g. 64) must use 3, not 64, when
        # computing its centre, otherwise the result is nowhere near the patch.
        # stride=4, i=2, Ni=3 (clipped) -> occupies global indices 8,9,10 ->
        # physical 80,90,100 -> true centre = 90.
        center_correct = c.generate_patch_center(2, 0, 0, 3, 1, 1, stride=4,
                                                   voxel_side=10.0, box_length=1000.0)
        self.assertAlmostEqual(float(center_correct[0]), 90.0)

        # Using the wrong (max) size of, say, 64 here would give
        # 2*4 + (64-1)/2 = 8 + 31.5 = 39.5 -> physical 395, far outside the patch.
        center_wrong = c.generate_patch_center(2, 0, 0, 64, 1, 1, stride=4,
                                                voxel_side=10.0, box_length=1000.0)
        self.assertNotAlmostEqual(float(center_wrong[0]), 90.0)


class TestComputeNPatches(unittest.TestCase):
    def test_docstring_example(self):
        # N=64, overlap=16 -> stride=48, N_global=79 (from the module docstring
        # example): (79-64)/48 = 0.3125 -> ceil = 1 -> n = 1+1 = 2.
        self.assertEqual(c.compute_n_patches(N_global=79, N=64, stride=48), 2)

    def test_single_patch_when_box_fits_in_one_patch(self):
        self.assertEqual(c.compute_n_patches(N_global=50, N=64, stride=48), 1)
        self.assertEqual(c.compute_n_patches(N_global=64, N=64, stride=48), 1)

    def test_exact_multiple_of_stride(self):
        # N_global=160, N=64, stride=48: (160-64)/48 = 2.0 -> ceil=2 -> n=3.
        # Check: last origin = 2*48=96, last size=min(64,160-96)=64,
        # 96+64=160 == N_global, so the box is exactly covered with no excess patch.
        n = c.compute_n_patches(N_global=160, N=64, stride=48)
        self.assertEqual(n, 3)
        last_origin = (n - 1) * 48
        last_size = min(64, 160 - last_origin)
        self.assertEqual(last_origin + last_size, 160)

    def test_always_covers_the_axis_for_many_configs(self):
        # Brute-force coverage check (deterministically seeded) for the formula
        # used in generate_patches_for_sim.  Each patch must reach N_global, and
        # one fewer patch must NOT reach N_global (minimality).
        rng = np.random.default_rng(0)
        for _ in range(2000):
            N = int(rng.integers(8, 100))
            overlap = int(rng.integers(0, N))
            stride = N - overlap
            N_global = int(rng.integers(N, N * 6))

            n = c.compute_n_patches(N_global, N, stride)
            last_origin = (n - 1) * stride
            last_size = min(N, N_global - last_origin)
            self.assertGreaterEqual(last_origin + last_size, N_global)

            if n > 1:
                prev_origin = (n - 2) * stride
                prev_size = min(N, N_global - prev_origin)
                self.assertLess(prev_origin + prev_size, N_global)


class TestAssignHalosToGlobalVoxels(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("conversion.test")

    def test_one_halo_per_voxel_exact_assignment(self):
        # box_length=4, N_global=2, voxel_side=2 -> 8 voxels at coords {0,2}^3.
        # Place one halo exactly at each voxel centre with a distinct velocity, so
        # the correct assignment (disp=0, vel=that halo's velocity) is obvious by
        # construction and easy to check by hand.
        box_length = 4.0
        global_vc = c.build_global_voxel_grid(2, 2.0, box_length)  # (2,2,2,3)
        pos = global_vc.reshape(8, 3).copy()
        vel = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)

        disp, vel_field = c.assign_halos_to_global_voxels(
            pos, vel, global_vc, box_length, self.logger
        )
        # disp shape (3,2,2,2); every voxel's halo is exactly at its own centre,
        # so displacement must be zero everywhere.
        np.testing.assert_allclose(disp, np.zeros((3, 2, 2, 2)), atol=1e-5)

        # Each voxel's assigned velocity must equal that voxel's own halo velocity
        # (since each halo coincides with exactly one voxel centre, it is the
        # unique nearest candidate for that voxel).
        vel_flat = vel_field.transpose(1, 2, 3, 0).reshape(8, 3)
        np.testing.assert_allclose(np.sort(vel_flat, axis=0), np.sort(vel, axis=0))

    def test_fewer_halos_than_voxels_zero_fills_the_rest(self):
        # 8 voxels, only 3 halos -> exactly 3 voxels get a nonzero assignment,
        # the remaining 5 stay zero-filled.
        box_length = 4.0
        global_vc = c.build_global_voxel_grid(2, 2.0, box_length)
        pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
        vel = np.array([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]], dtype=np.float32)

        disp, vel_field = c.assign_halos_to_global_voxels(
            pos, vel, global_vc, box_length, self.logger
        )
        vel_flat = vel_field.transpose(1, 2, 3, 0).reshape(8, 3)
        n_nonzero = int(np.sum(np.any(vel_flat != 0, axis=1)))
        self.assertEqual(n_nonzero, 3)


class TestMakeWeightWindow(unittest.TestCase):
    def test_hann_4_matches_hand_computed_values(self):
        # np.hanning(4) = 0.5 - 0.5*cos(2*pi*n/3) for n=0..3
        #   n=0: 0.5 - 0.5*cos(0)        = 0
        #   n=1: 0.5 - 0.5*cos(2pi/3)    = 0.5 - 0.5*(-0.5) = 0.75
        #   n=2: same as n=1 by symmetry = 0.75
        #   n=3: same as n=0             = 0
        # floored at eps=1e-3 -> [0.001, 0.75, 0.75, 0.001]
        w = c.make_weight_window(4, 1, 1, mode="hann", eps=1e-3)
        np.testing.assert_allclose(w[0, :, 0, 0], [0.001, 0.75, 0.75, 0.001], atol=1e-6)

    def test_uniform_is_all_ones(self):
        w = c.make_weight_window(3, 2, 4, mode="uniform")
        self.assertEqual(w.shape, (1, 3, 2, 4))
        np.testing.assert_allclose(w, np.ones((1, 3, 2, 4)))


class TestAddPatchPeriodic(unittest.TestCase):
    def test_wrap_around_accumulates_into_correct_voxels(self):
        # global_N=4; a 2-voxel patch starting at origin (3, 0, 0) must wrap its
        # second x-index from 4 to 0 (i.e. covers global x = 3 and x = 0).
        accum = np.zeros((1, 4, 4, 4), dtype=np.float32)
        weights = np.zeros((1, 4, 4, 4), dtype=np.float32)
        patch = np.array([[[[10.0]], [[20.0]]]], dtype=np.float32)  # shape (1,2,1,1)
        window = np.ones((1, 2, 1, 1), dtype=np.float32)

        c.add_patch_periodic(accum, weights, patch, origin=(3, 0, 0), window=window)

        self.assertAlmostEqual(float(accum[0, 3, 0, 0]), 10.0)
        self.assertAlmostEqual(float(accum[0, 0, 0, 0]), 20.0)
        self.assertAlmostEqual(float(weights[0, 3, 0, 0]), 1.0)
        self.assertAlmostEqual(float(weights[0, 0, 0, 0]), 1.0)


class TestComputeNGlobal(unittest.TestCase):
    def test_rounds_up_where_round_would_undercount(self):
        # M=800 halos, L=100: voxel_side = (1e6/800)^(1/3), so
        # L/voxel_side = 800^(1/3) ≈ 9.283.  round() -> 9 -> 9^3 = 729 < 800
        # voxels (71 halos could never be assigned); ceil -> 10 -> 1000 >= 800.
        M, L = 800, 100.0
        voxel_side = (L ** 3 / M) ** (1.0 / 3.0)
        self.assertEqual(c.compute_n_global(L, voxel_side), 10)

    def test_perfect_cube_halo_count_is_exact(self):
        # M=1000, L=100: L/voxel_side = 1000^(1/3) = 10 exactly; the epsilon
        # guard must keep this at 10, not over-allocate to 11.
        M, L = 1000, 100.0
        voxel_side = (L ** 3 / M) ** (1.0 / 3.0)
        self.assertEqual(c.compute_n_global(L, voxel_side), 10)

    def test_grid_capacity_always_at_least_halo_count(self):
        # The invariant that motivated the ceil: N_global^3 >= M for any halo
        # count, so assign_halos_to_global_voxels (one halo per voxel) can
        # always place every halo.
        rng = np.random.default_rng(0)
        L = 1000.0
        for M in rng.integers(2, 2_000_000, size=500):
            M = int(M)
            voxel_side = (L ** 3 / M) ** (1.0 / 3.0)
            N_global = c.compute_n_global(L, voxel_side)
            self.assertGreaterEqual(
                N_global ** 3, M,
                msg=f"M={M}: N_global={N_global} gives only {N_global**3} voxels",
            )


class TestHaloCountsField(unittest.TestCase):
    def test_docstring_example_by_hand(self):
        # box=4, voxel_side=2, N_global=2 -> nodes at {0, 2} per axis.
        # (0.9,0,0): round(0.45)=0            -> voxel (0,0,0)
        # (1.1,0,0): round(0.55)=1            -> voxel (1,0,0)
        # (3.1,3.1,3.1): round(1.55)=2 mod 2  -> voxel (0,0,0)  (periodic wrap:
        #     node 0 is 0.9 away through the boundary, node 2 is 1.1 away)
        # (0,0,0): exactly on node 0          -> voxel (0,0,0)
        pos = np.array([
            [0.9, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [3.1, 3.1, 3.1],
            [0.0, 0.0, 0.0],
        ], dtype=np.float32)
        counts = c.halo_counts_field(pos, voxel_side=2.0, N_global=2, box_length=4.0)

        self.assertEqual(counts.shape, (1, 2, 2, 2))
        self.assertAlmostEqual(float(counts[0, 0, 0, 0]), 3.0)
        self.assertAlmostEqual(float(counts[0, 1, 0, 0]), 1.0)
        # Every halo lands in exactly one voxel: total == M.
        self.assertAlmostEqual(float(counts.sum()), 4.0)

    def test_halos_on_nodes_count_in_their_own_voxel(self):
        # Consistency with the displacement grid: a halo exactly at a lattice
        # node (from build_global_voxel_grid) must be counted at that node.
        box_length = 4.0
        global_vc = c.build_global_voxel_grid(2, 2.0, box_length)   # (2,2,2,3)
        pos = global_vc.reshape(8, 3).copy()
        counts = c.halo_counts_field(pos, 2.0, 2, box_length)
        np.testing.assert_allclose(counts[0], np.ones((2, 2, 2)))

    def test_multiplicity_is_preserved_not_capped_at_one(self):
        # Three halos in the same voxel -> count 3 (the one-to-one assignment
        # in assign_halos_to_global_voxels would spread these over 3 voxels).
        pos = np.array([[0.1, 0.1, 0.1]] * 3, dtype=np.float32)
        counts = c.halo_counts_field(pos, 2.0, 2, 4.0)
        self.assertAlmostEqual(float(counts[0, 0, 0, 0]), 3.0)


class TestStitchRoundTrip(unittest.TestCase):
    def test_generate_style_slicing_then_stitch_recovers_global_field(self):
        # End-to-end: slice a known global field into overlapping patches with
        # the exact layout of generate_patches_for_sim, write them (plus
        # meta.json) to disk, stitch, and demand exact recovery.
        # N_global=10, N=4, overlap=2 -> stride=2,
        # n_patches = ceil((10-4)/2)+1 = 4, patch origins {0,2,4,6}.
        N_global, N, overlap, sim_id = 10, 4, 2, 0
        stride = N - overlap
        n_patches = c.compute_n_patches(N_global, N, stride)
        self.assertEqual(n_patches, 4)

        rng = np.random.default_rng(1)
        gdisp = rng.standard_normal((3, N_global, N_global, N_global)).astype(np.float32)
        gvel  = rng.standard_normal((3, N_global, N_global, N_global)).astype(np.float32)
        ch = np.arange(3)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            catalog_dir = root / f"quijote-{N}"

            for i in range(n_patches):
                for j in range(n_patches):
                    for k in range(n_patches):
                        Ni = min(N, N_global - i * stride)
                        Nj = min(N, N_global - j * stride)
                        Nk = min(N, N_global - k * stride)
                        xi = np.arange(i * stride, i * stride + Ni) % N_global
                        yj = np.arange(j * stride, j * stride + Nj) % N_global
                        zk = np.arange(k * stride, k * stride + Nk) % N_global
                        out = catalog_dir / f"set{sim_id}_pos_{i}_{j}_{k}" / "PART_009"
                        c.save_outputs(
                            out,
                            gdisp[np.ix_(ch, xi, yj, zk)],
                            gvel [np.ix_(ch, xi, yj, zk)],
                            None, skip_existing=False,
                        )

            catalog_dir.mkdir(parents=True, exist_ok=True)
            with open(catalog_dir / f"set{sim_id}_meta.json", "w") as f:
                json.dump({"N_global": N_global, "N": N, "overlap": overlap,
                           "stride": stride, "n_patches": n_patches}, f)

            stitched_dir = c.stitch_patches(
                root, root, sim_id, "quijote", N, weight_mode="uniform"
            )

            # All patches are slices of the SAME global field, so any weighted
            # average must reproduce it exactly (up to float32 arithmetic).
            np.testing.assert_allclose(
                np.load(stitched_dir / "disp.npy"), gdisp, atol=1e-5
            )
            np.testing.assert_allclose(
                np.load(stitched_dir / "vel.npy"), gvel, atol=1e-5
            )


class TestDispToRhoLatticeConvention(unittest.TestCase):
    """Pins the node-centred lattice convention shared by conversion.py and
    the density post-processing (regression tests for the half-cell offset)."""

    def test_zero_displacement_gives_uniform_density(self):
        # disp = 0 -> every particle sits exactly on a lattice node -> each
        # deposit cell receives exactly its own particle -> rho == 1 everywhere.
        disp = np.zeros((3, 4, 4, 4), dtype=np.float32)
        rho = dh.disp_to_rho(disp, box_size=8.0)
        np.testing.assert_allclose(rho, np.ones((4, 4, 4)), atol=1e-6)

    def test_single_half_cell_displacement_by_hand(self):
        # N=2, box=4, cell=2, nodes at {0, 2}.  Displace only the particle at
        # node (0,0,0) by +1 along x: it lands at x=1 -> cell coordinate 0.5 ->
        # CIC splits it 0.5/0.5 between cells 0 and 1 along x.
        #   rho[0,0,0] = 0.5                      (half of the displaced particle)
        #   rho[1,0,0] = 1 + 0.5 = 1.5            (own particle + other half)
        #   all other cells = 1                   (their particles undisplaced)
        # The old cell-centred lattice ((i+0.5)*cell) would have started this
        # particle at x=1 and landed it at x=2, giving rho[1,0,0]=2 instead.
        disp = np.zeros((3, 2, 2, 2), dtype=np.float32)
        disp[0, 0, 0, 0] = 1.0
        rho = dh.disp_to_rho(disp, box_size=4.0)

        self.assertAlmostEqual(float(rho[0, 0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(rho[1, 0, 0]), 1.5, places=5)
        self.assertAlmostEqual(float(rho[0, 1, 1]), 1.0, places=5)
        self.assertAlmostEqual(float(rho.sum()), 8.0, places=4)

    def test_reconstruction_matches_direct_halo_deposit(self):
        # The real invariant behind the offset fix: if disp encodes halo
        # positions relative to the lattice nodes (exactly how conversion.py
        # builds it), then depositing the displaced lattice must give the SAME
        # density as depositing the halos directly.  Uses a non-uniform grid
        # (N * voxel_side != box) to also exercise the voxel_side parameter.
        N, box, voxel_side = 4, 10.0, 2.3   # nodes at {0, 2.3, 4.6, 6.9}
        rng = np.random.default_rng(2)

        g = np.arange(N) * voxel_side
        nodes = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=0)  # (3,N,N,N)
        jitter = rng.uniform(-0.6, 0.6, size=(3, N, N, N))
        halos = ((nodes + jitter) % box).reshape(3, -1).T               # (N^3, 3)

        # disp exactly as conversion.py defines it: halo - node (min image).
        disp = c.periodic_disp_vec(halos.T.reshape(3, N, N, N), nodes, box)

        rho_from_disp  = dh.disp_to_rho(disp.astype(np.float32), box,
                                        voxel_side=voxel_side)
        rho_from_halos = dh.halos_to_rho(halos.astype(np.float32), box, N)
        np.testing.assert_allclose(rho_from_disp, rho_from_halos, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
