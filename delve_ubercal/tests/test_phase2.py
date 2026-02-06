"""Unit tests for Phase 2: CG Sparse Solver.

The critical test is the synthetic data recovery test: generate fake
observations with known zero-points, add noise, solve, verify recovery.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from delve_ubercal.phase0_ingest import load_config
from delve_ubercal.phase2_solve import (
    accumulate_normal_equations,
    build_node_index,
    solve_anchored,
    solve_unanchored,
)


@pytest.fixture
def config():
    return load_config()


def make_synthetic_data(
    n_ccd_exposures=1000,
    n_stars=5000,
    n_des=200,
    noise_mmag=5.0,
    seed=42,
):
    """Generate synthetic star observations with known zero-points.

    Parameters
    ----------
    n_ccd_exposures : int
        Number of (expnum, ccdnum) nodes.
    n_stars : int
        Number of unique stars.
    n_des : int
        Number of CCD-exposures that are "DES" (for anchoring).
    noise_mmag : float
        Noise in mmag added to each detection.
    seed : int
        Random seed.

    Returns
    -------
    star_list_file : Path
        Path to parquet file with detections.
    connected_file : Path
        Path to connected nodes parquet.
    fgcm_file : Path
        Path to DES FGCM parquet.
    true_zps : np.ndarray
        True zero-points (length n_ccd_exposures).
    node_list : list of (expnum, ccdnum)
        Node definitions.
    tmpdir : Path
        Temporary directory containing all files.
    """
    rng = np.random.default_rng(seed)
    tmpdir = Path(tempfile.mkdtemp())

    # Define CCD-exposures
    node_list = [(1000 + i, (i % 62) + 1) for i in range(n_ccd_exposures)]

    # True zero-points: random values centered around 25 mag with ~0.1 mag scatter
    true_zps = 25.0 + rng.normal(0, 0.1, n_ccd_exposures)

    # True star magnitudes
    true_mags = rng.uniform(17.0, 20.0, n_stars)

    # Generate detections: each star observed on 3-10 random CCD-exposures
    records = []
    for s in range(n_stars):
        n_obs = rng.integers(3, min(11, n_ccd_exposures + 1))
        obs_indices = rng.choice(n_ccd_exposures, size=n_obs, replace=False)
        for idx in obs_indices:
            expnum, ccdnum = node_list[idx]
            # m_inst = m_true - ZP_true + noise
            # (m_cal = m_inst + ZP => m_true = m_inst + ZP => m_inst = m_true - ZP)
            noise = rng.normal(0, noise_mmag / 1000.0)
            m_inst = true_mags[s] - true_zps[idx] + noise
            m_err = noise_mmag / 1000.0  # constant error
            records.append({
                "objectid": f"star_{s}",
                "expnum": expnum,
                "ccdnum": ccdnum,
                "m_inst": m_inst,
                "m_err": m_err,
            })

    df = pd.DataFrame(records)

    # Save star list
    phase1_dir = tmpdir / "phase1_g"
    phase1_dir.mkdir(parents=True, exist_ok=True)
    star_file = phase1_dir / "star_lists_nside32_pixel0.parquet"
    df.to_parquet(star_file, index=False)

    # Save connected nodes
    connected_df = pd.DataFrame(node_list, columns=["expnum", "ccdnum"])
    connected_df["expnum"] = connected_df["expnum"].astype(np.int64)
    connected_df["ccdnum"] = connected_df["ccdnum"].astype(np.int32)
    connected_file = phase1_dir / "connected_nodes.parquet"
    connected_df.to_parquet(connected_file, index=False)

    # Save DES FGCM (first n_des CCD-exposures are "DES")
    cache_dir = tmpdir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    des_records = []
    for i in range(n_des):
        expnum, ccdnum = node_list[i]
        des_records.append({
            "expnum": expnum,
            "ccdnum": ccdnum,
            "band": "g",
            "mag_zero": true_zps[i],  # FGCM matches true ZP for DES
            "sigma_mag_zero": 0.001,
            "source": "FGCM",
            "flag": 0,
        })
    fgcm_df = pd.DataFrame(des_records)
    fgcm_file = cache_dir / "des_y6_fgcm_zeropoints.parquet"
    fgcm_df.to_parquet(fgcm_file, index=False)

    return star_file, connected_file, fgcm_file, true_zps, node_list, tmpdir


class TestNormalEquationsAccumulation:
    """Test the normal equations matrix construction."""

    def test_simple_two_node(self):
        """Two nodes, one star: verify matrix structure."""
        tmpdir = Path(tempfile.mkdtemp())

        df = pd.DataFrame({
            "objectid": ["s1", "s1"],
            "expnum": [100, 200],
            "ccdnum": [1, 1],
            "m_inst": [18.0, 18.1],
            "m_err": [0.01, 0.01],
        })
        f = tmpdir / "stars.parquet"
        df.to_parquet(f, index=False)

        node_to_idx = {(100, 1): 0, (200, 1): 1}
        AtWA, rhs, n_stars, n_pairs = accumulate_normal_equations(
            [f], node_to_idx, 2
        )

        assert n_stars == 1
        assert n_pairs == 1

        # Weight: 1/(0.01^2 + 0.01^2) = 5000
        w = 1.0 / (0.01**2 + 0.01**2)
        assert abs(AtWA[0, 0] - w) < 1e-6
        assert abs(AtWA[1, 1] - w) < 1e-6
        assert abs(AtWA[0, 1] + w) < 1e-6  # negative off-diagonal
        assert abs(AtWA[1, 0] + w) < 1e-6

        # RHS: -w * (ma - mb) for node 0, +w * (ma - mb) for node 1
        dm = 18.0 - 18.1
        assert abs(rhs[0] + w * dm) < 1e-6  # -w*dm
        assert abs(rhs[1] - w * dm) < 1e-6  # +w*dm

    def test_symmetric(self):
        """Normal equations matrix should be symmetric."""
        tmpdir = Path(tempfile.mkdtemp())

        rng = np.random.default_rng(42)
        records = []
        for s in range(100):
            nodes = [(1000 + i, 1) for i in rng.choice(10, size=3, replace=False)]
            for expnum, ccdnum in nodes:
                records.append({
                    "objectid": f"s{s}",
                    "expnum": expnum,
                    "ccdnum": ccdnum,
                    "m_inst": rng.normal(18.0, 0.1),
                    "m_err": 0.01,
                })

        df = pd.DataFrame(records)
        f = tmpdir / "stars.parquet"
        df.to_parquet(f, index=False)

        node_to_idx = {(1000 + i, 1): i for i in range(10)}
        AtWA, rhs, _, _ = accumulate_normal_equations([f], node_to_idx, 10)

        # Check symmetry
        diff = AtWA - AtWA.T
        assert abs(diff).max() < 1e-10

    def test_graph_laplacian_null_space(self):
        """AtWA should be a graph Laplacian — constant vector in null space."""
        tmpdir = Path(tempfile.mkdtemp())

        # All mags identical => rhs = 0, AtWA * ones ~ 0
        records = []
        for s in range(50):
            for i in range(3):
                records.append({
                    "objectid": f"s{s}",
                    "expnum": 1000 + (s * 3 + i) % 5,
                    "ccdnum": 1,
                    "m_inst": 18.0,
                    "m_err": 0.01,
                })

        df = pd.DataFrame(records)
        f = tmpdir / "stars.parquet"
        df.to_parquet(f, index=False)

        node_to_idx = {(1000 + i, 1): i for i in range(5)}
        AtWA, rhs, _, _ = accumulate_normal_equations([f], node_to_idx, 5)

        # AtWA * [1,1,...,1] should be ~0 (graph Laplacian property)
        result = AtWA @ np.ones(5)
        assert np.max(np.abs(result)) < 1e-8


class TestSyntheticSolve:
    """Critical test: recover known zero-points from synthetic data."""

    def test_unanchored_recovery(self, config):
        """Unanchored solve should recover true ZPs to < 1 mmag RMS."""
        star_file, connected_file, fgcm_file, true_zps, node_list, tmpdir = \
            make_synthetic_data(n_ccd_exposures=200, n_stars=2000, n_des=50, seed=42)

        node_to_idx, idx_to_node = build_node_index(connected_file)
        n_params = len(node_to_idx)

        des_fgcm_zps = {}
        fgcm = pd.read_parquet(fgcm_file)
        for _, row in fgcm.iterrows():
            des_fgcm_zps[(row["expnum"], row["ccdnum"])] = row["mag_zero"]

        AtWA, rhs, n_stars, n_pairs = accumulate_normal_equations(
            [star_file], node_to_idx, n_params
        )

        zp_solved, info = solve_unanchored(
            AtWA, rhs, config, node_to_idx, des_fgcm_zps
        )

        assert info["converged"]

        # Compare to true ZPs (reorder to match solved indices)
        true_ordered = np.array([
            true_zps[node_list.index(node)] for node in idx_to_node
        ])
        residual = zp_solved - true_ordered
        rms_mmag = np.sqrt(np.mean(residual**2)) * 1000

        print(f"Unanchored recovery RMS: {rms_mmag:.3f} mmag")
        assert rms_mmag < 1.0, f"RMS {rms_mmag:.3f} mmag > 1 mmag"

    def test_anchored_recovery(self, config):
        """Anchored solve should recover true ZPs to < 1 mmag RMS."""
        star_file, connected_file, fgcm_file, true_zps, node_list, tmpdir = \
            make_synthetic_data(n_ccd_exposures=200, n_stars=2000, n_des=50, seed=42)

        node_to_idx, idx_to_node = build_node_index(connected_file)
        n_params = len(node_to_idx)

        des_fgcm_zps = {}
        fgcm = pd.read_parquet(fgcm_file)
        for _, row in fgcm.iterrows():
            des_fgcm_zps[(row["expnum"], row["ccdnum"])] = row["mag_zero"]

        AtWA, rhs, n_stars, n_pairs = accumulate_normal_equations(
            [star_file], node_to_idx, n_params
        )

        zp_solved, info = solve_anchored(
            AtWA, rhs, config, node_to_idx, des_fgcm_zps
        )

        assert info["converged"]

        true_ordered = np.array([
            true_zps[node_list.index(node)] for node in idx_to_node
        ])
        residual = zp_solved - true_ordered
        rms_mmag = np.sqrt(np.mean(residual**2)) * 1000

        print(f"Anchored recovery RMS: {rms_mmag:.3f} mmag")
        assert rms_mmag < 1.0, f"RMS {rms_mmag:.3f} mmag > 1 mmag"

    def test_anchored_des_pinned(self, config):
        """In anchored mode, DES CCD-exposures should be pinned to FGCM."""
        star_file, connected_file, fgcm_file, true_zps, node_list, tmpdir = \
            make_synthetic_data(n_ccd_exposures=200, n_stars=2000, n_des=50, seed=42)

        node_to_idx, idx_to_node = build_node_index(connected_file)
        n_params = len(node_to_idx)

        des_fgcm_zps = {}
        fgcm = pd.read_parquet(fgcm_file)
        for _, row in fgcm.iterrows():
            des_fgcm_zps[(row["expnum"], row["ccdnum"])] = row["mag_zero"]

        AtWA, rhs, _, _ = accumulate_normal_equations(
            [star_file], node_to_idx, n_params
        )

        zp_solved, info = solve_anchored(
            AtWA, rhs, config, node_to_idx, des_fgcm_zps
        )

        # DES nodes should be very close to FGCM values
        # With anchor_weight=1e6 and 5 mmag noise, allow up to 2 mmag
        for node, idx in node_to_idx.items():
            if node in des_fgcm_zps:
                diff_mmag = abs(zp_solved[idx] - des_fgcm_zps[node]) * 1000
                assert diff_mmag < 2.0, \
                    f"DES node {node}: {diff_mmag:.3f} mmag from FGCM"

    def test_larger_synthetic(self, config):
        """Larger test: 1000 CCD-exposures, 5000 stars."""
        star_file, connected_file, fgcm_file, true_zps, node_list, tmpdir = \
            make_synthetic_data(
                n_ccd_exposures=1000, n_stars=5000, n_des=200, seed=123
            )

        node_to_idx, idx_to_node = build_node_index(connected_file)
        n_params = len(node_to_idx)

        des_fgcm_zps = {}
        fgcm = pd.read_parquet(fgcm_file)
        for _, row in fgcm.iterrows():
            des_fgcm_zps[(row["expnum"], row["ccdnum"])] = row["mag_zero"]

        AtWA, rhs, n_stars, n_pairs = accumulate_normal_equations(
            [star_file], node_to_idx, n_params
        )

        # Unanchored
        zp_un, info_un = solve_unanchored(
            AtWA, rhs, config, node_to_idx, des_fgcm_zps
        )
        assert info_un["converged"]

        true_ordered = np.array([
            true_zps[node_list.index(node)] for node in idx_to_node
        ])
        rms_un = np.sqrt(np.mean((zp_un - true_ordered)**2)) * 1000
        print(f"Large synthetic unanchored RMS: {rms_un:.3f} mmag")
        assert rms_un < 2.0, f"Large unanchored RMS {rms_un:.3f} mmag > 2 mmag"

        # Anchored
        zp_an, info_an = solve_anchored(
            AtWA, rhs, config, node_to_idx, des_fgcm_zps
        )
        assert info_an["converged"]

        rms_an = np.sqrt(np.mean((zp_an - true_ordered)**2)) * 1000
        print(f"Large synthetic anchored RMS: {rms_an:.3f} mmag")
        assert rms_an < 2.0, f"Large anchored RMS {rms_an:.3f} mmag > 2 mmag"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
