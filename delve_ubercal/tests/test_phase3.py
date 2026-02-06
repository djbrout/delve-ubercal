"""Unit tests for Phase 3: Iterative Outlier Rejection."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from delve_ubercal.phase0_ingest import load_config
from delve_ubercal.phase2_solve import (
    accumulate_normal_equations,
    build_node_index,
    solve_anchored,
    solve_unanchored,
)
from delve_ubercal.phase3_outlier_rejection import (
    compute_residuals,
    flag_bad_ccds,
    flag_bad_exposures,
    flag_outlier_detections,
    flag_variable_stars,
)


@pytest.fixture
def config():
    return load_config()


def make_synthetic_with_outliers(
    n_ccd_exposures=200,
    n_stars=500,
    n_des=50,
    noise_mmag=5.0,
    n_variable_stars=20,
    n_outlier_dets=50,
    n_bad_exposures=3,
    seed=42,
):
    """Generate synthetic data with known outliers.

    Returns
    -------
    star_file : Path
    connected_file : Path
    fgcm_file : Path
    true_zps : np.ndarray
    node_list : list of (expnum, ccdnum)
    variable_star_ids : set
    bad_exposure_ids : set
    tmpdir : Path
    """
    rng = np.random.default_rng(seed)
    tmpdir = Path(tempfile.mkdtemp())

    node_list = [(1000 + i, (i % 62) + 1) for i in range(n_ccd_exposures)]
    true_zps = 25.0 + rng.normal(0, 0.1, n_ccd_exposures)
    true_mags = rng.uniform(17.0, 20.0, n_stars)

    # Assign stars as variable (random large scatter)
    variable_star_ids = set()
    for s in range(n_variable_stars):
        variable_star_ids.add(f"star_{s}")

    # Assign bad exposures (large ZP offset from nightly median)
    bad_exp_indices = rng.choice(
        range(n_des, n_ccd_exposures),  # Don't make DES exposures bad
        size=n_bad_exposures, replace=False,
    )
    bad_exposure_ids = set()
    for idx in bad_exp_indices:
        true_zps[idx] += 0.5  # 500 mmag offset
        bad_exposure_ids.add(node_list[idx][0])

    records = []
    for s in range(n_stars):
        n_obs = rng.integers(3, min(11, n_ccd_exposures + 1))
        obs_indices = rng.choice(n_ccd_exposures, size=n_obs, replace=False)
        star_id = f"star_{s}"

        for idx in obs_indices:
            expnum, ccdnum = node_list[idx]
            if star_id in variable_star_ids:
                # Variable star: add large extra scatter (50 mmag)
                noise = rng.normal(0, 50.0 / 1000.0)
            else:
                noise = rng.normal(0, noise_mmag / 1000.0)
            m_inst = true_mags[s] - true_zps[idx] + noise
            m_err = noise_mmag / 1000.0
            records.append({
                "objectid": star_id,
                "expnum": expnum,
                "ccdnum": ccdnum,
                "m_inst": m_inst,
                "m_err": m_err,
            })

    df = pd.DataFrame(records)

    # Save files
    phase1_dir = tmpdir / "phase1_g"
    phase1_dir.mkdir(parents=True, exist_ok=True)
    star_file = phase1_dir / "star_lists_nside32_pixel0.parquet"
    df.to_parquet(star_file, index=False)

    connected_df = pd.DataFrame(node_list, columns=["expnum", "ccdnum"])
    connected_df["expnum"] = connected_df["expnum"].astype(np.int64)
    connected_df["ccdnum"] = connected_df["ccdnum"].astype(np.int32)
    connected_file = phase1_dir / "connected_nodes.parquet"
    connected_df.to_parquet(connected_file, index=False)

    cache_dir = tmpdir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    des_records = []
    for i in range(n_des):
        expnum, ccdnum = node_list[i]
        des_records.append({
            "expnum": expnum, "ccdnum": ccdnum, "band": "g",
            "mag_zero": true_zps[i], "sigma_mag_zero": 0.001,
            "source": "FGCM", "flag": 0,
        })
    fgcm_df = pd.DataFrame(des_records)
    fgcm_file = cache_dir / "des_y6_fgcm_zeropoints.parquet"
    fgcm_df.to_parquet(fgcm_file, index=False)

    return (star_file, connected_file, fgcm_file, true_zps, node_list,
            variable_star_ids, bad_exposure_ids, tmpdir)


class TestResidualComputation:
    """Test residual computation."""

    def test_perfect_data_zero_residuals(self):
        """With perfect ZPs, residuals should be near zero."""
        tmpdir = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(42)

        true_zps = [25.0, 25.1]
        true_mag = 18.0
        records = []
        for i, (zp, expnum) in enumerate(zip(true_zps, [100, 200])):
            m_inst = true_mag - zp
            records.append({
                "objectid": "s1", "expnum": expnum, "ccdnum": 1,
                "m_inst": m_inst, "m_err": 0.005,
            })

        df = pd.DataFrame(records)
        f = tmpdir / "stars.parquet"
        df.to_parquet(f, index=False)

        node_to_idx = {(100, 1): 0, (200, 1): 1}
        zp_dict = {(100, 1): 25.0, (200, 1): 25.1}

        residuals = compute_residuals([f], zp_dict, node_to_idx)
        assert len(residuals) == 2
        assert np.all(np.abs(residuals["residual"].values) < 1e-10)

    def test_nonzero_residuals(self):
        """With wrong ZPs, residuals should be nonzero."""
        tmpdir = Path(tempfile.mkdtemp())
        records = [
            {"objectid": "s1", "expnum": 100, "ccdnum": 1,
             "m_inst": -7.0, "m_err": 0.005},
            {"objectid": "s1", "expnum": 200, "ccdnum": 1,
             "m_inst": -7.1, "m_err": 0.005},
        ]
        df = pd.DataFrame(records)
        f = tmpdir / "stars.parquet"
        df.to_parquet(f, index=False)

        node_to_idx = {(100, 1): 0, (200, 1): 1}
        # Wrong ZPs (both 25.0 instead of 25.0 and 25.1)
        zp_dict = {(100, 1): 25.0, (200, 1): 25.0}

        residuals = compute_residuals([f], zp_dict, node_to_idx)
        # m_cal = [-7.0+25.0, -7.1+25.0] = [18.0, 17.9]
        # m_star_mean ~ 17.95 (weighted)
        assert np.any(np.abs(residuals["residual"].values) > 0.01)


class TestFlagVariableStars:
    """Test variable star flagging."""

    def test_flags_high_scatter_star(self):
        """Star with large scatter should be flagged."""
        records = []
        # Normal star: low scatter
        for i in range(5):
            records.append({
                "objectid": "normal", "expnum": 100 + i, "ccdnum": 1,
                "m_inst": 0.0, "m_err": 0.005,
                "m_cal": 18.0 + np.random.normal(0, 0.003),
                "m_star_mean": 18.0,
                "residual": np.random.normal(0, 0.003),
                "zp_solved": 25.0, "n_det": 5,
            })
        # Variable star: large scatter
        for i in range(5):
            r = np.random.normal(0, 0.05)  # 50 mmag scatter
            records.append({
                "objectid": "variable", "expnum": 100 + i, "ccdnum": 1,
                "m_inst": 0.0, "m_err": 0.005,
                "m_cal": 18.0 + r,
                "m_star_mean": 18.0,
                "residual": r,
                "zp_solved": 25.0, "n_det": 5,
            })

        df = pd.DataFrame(records)
        flagged = flag_variable_stars(df, chi2_cut=3.0)
        assert "variable" in flagged
        assert "normal" not in flagged

    def test_no_false_positives_for_good_stars(self):
        """Stars with noise consistent with errors should not be flagged."""
        rng = np.random.default_rng(42)
        records = []
        for s in range(50):
            for i in range(5):
                r = rng.normal(0, 0.005)  # Consistent with m_err=0.005
                records.append({
                    "objectid": f"s{s}", "expnum": 100 + i, "ccdnum": 1,
                    "m_inst": 0.0, "m_err": 0.005,
                    "m_cal": 18.0 + r, "m_star_mean": 18.0,
                    "residual": r, "zp_solved": 25.0, "n_det": 5,
                })
        df = pd.DataFrame(records)
        flagged = flag_variable_stars(df, chi2_cut=3.0)
        # Should flag < 10% of stars (chi2/dof > 3 for well-behaved stars is rare)
        assert len(flagged) < 10


class TestFlagOutlierDetections:
    """Test individual detection flagging."""

    def test_flags_large_outliers(self):
        """Detections with |residual| > 5*sigma should be flagged."""
        df = pd.DataFrame({
            "residual": [0.001, 0.002, 0.05, -0.003, 0.001],
            "m_err": [0.005, 0.005, 0.005, 0.005, 0.005],
        })
        mask = flag_outlier_detections(df, sigma_cut=5.0)
        assert mask[2]  # 0.05 / 0.005 = 10 > 5
        assert not mask[0]
        assert not mask[1]

    def test_respects_per_detection_errors(self):
        """Flagging should use each detection's own error."""
        df = pd.DataFrame({
            "residual": [0.025, 0.025],
            "m_err": [0.01, 0.004],  # First: 2.5 sigma, second: 6.25 sigma
        })
        mask = flag_outlier_detections(df, sigma_cut=5.0)
        assert not mask[0]
        assert mask[1]


class TestFlagBadExposures:
    """Test bad exposure flagging."""

    def test_flags_offset_exposure(self):
        """Exposure with ZP far from nightly median should be flagged."""
        zp_df = pd.DataFrame({
            "expnum": [100, 101, 102, 103, 200],
            "ccdnum": [1, 1, 1, 1, 1],
            "zp_solved": [25.0, 25.01, 24.99, 25.02, 25.5],
        })
        # All on same night except 200 which is offset
        mjds = {100: 56000.1, 101: 56000.2, 102: 56000.3,
                103: 56000.4, 200: 56000.5}
        flagged = flag_bad_exposures(zp_df, mjds, zp_cut=0.3)
        assert 200 in flagged

    def test_no_false_positives(self):
        """Normal exposures should not be flagged."""
        zp_df = pd.DataFrame({
            "expnum": [100, 101, 102],
            "ccdnum": [1, 1, 1],
            "zp_solved": [25.0, 25.01, 24.99],
        })
        mjds = {100: 56000.1, 101: 56000.2, 102: 56000.3}
        flagged = flag_bad_exposures(zp_df, mjds, zp_cut=0.3)
        assert len(flagged) == 0


class TestFlagBadCCDs:
    """Test bad CCD flagging."""

    def test_flags_high_scatter_ccd(self):
        """CCD with anomalously large scatter should be flagged."""
        rng = np.random.default_rng(42)
        records = []
        # 10 normal CCDs
        for c in range(10):
            for s in range(20):
                records.append({
                    "expnum": 100, "ccdnum": c + 1,
                    "residual": rng.normal(0, 0.005),
                })
        # 1 bad CCD
        for s in range(20):
            records.append({
                "expnum": 100, "ccdnum": 99,
                "residual": rng.normal(0, 0.1),  # 20x worse
            })
        df = pd.DataFrame(records)
        flagged = flag_bad_ccds(df, sigma_factor=3.0)
        assert (100, 99) in flagged


class TestSyntheticOutlierRejection:
    """End-to-end test: synthetic data with injected outliers."""

    def test_variable_stars_improve_rms(self, config):
        """Removing variable stars should reduce residual RMS."""
        (star_file, connected_file, fgcm_file, true_zps, node_list,
         variable_star_ids, bad_exposure_ids, tmpdir) = \
            make_synthetic_with_outliers(
                n_ccd_exposures=200, n_stars=500, n_des=50,
                n_variable_stars=30, n_outlier_dets=0, n_bad_exposures=0,
                seed=42,
            )

        node_to_idx, idx_to_node = build_node_index(connected_file)
        n_params = len(node_to_idx)

        des_fgcm_zps = {}
        fgcm = pd.read_parquet(fgcm_file)
        for _, row in fgcm.iterrows():
            des_fgcm_zps[(row["expnum"], row["ccdnum"])] = row["mag_zero"]

        # Initial solve
        AtWA, rhs, _, _ = accumulate_normal_equations(
            [star_file], node_to_idx, n_params
        )
        zp_arr, info = solve_anchored(AtWA, rhs, config, node_to_idx, des_fgcm_zps)
        zp_dict = {node: zp_arr[idx] for node, idx in node_to_idx.items()}

        # Compute residuals and flag
        residuals = compute_residuals([star_file], zp_dict, node_to_idx)
        rms_before = np.sqrt(np.mean(residuals["residual"].values ** 2)) * 1000
        flagged_stars = flag_variable_stars(residuals, chi2_cut=3.0)

        # Remove flagged stars and re-solve
        clean_df = residuals[~residuals["objectid"].isin(flagged_stars)]
        clean_file = tmpdir / "clean.parquet"
        clean_df[["objectid", "expnum", "ccdnum", "m_inst", "m_err"]].to_parquet(
            clean_file, index=False
        )

        AtWA2, rhs2, _, _ = accumulate_normal_equations(
            [clean_file], node_to_idx, n_params
        )
        zp_arr2, info2 = solve_anchored(
            AtWA2, rhs2, config, node_to_idx, des_fgcm_zps
        )

        # Compare to true ZPs
        true_ordered = np.array([
            true_zps[node_list.index(node)] for node in idx_to_node
        ])
        rms_initial = np.sqrt(np.mean((zp_arr - true_ordered) ** 2)) * 1000
        rms_cleaned = np.sqrt(np.mean((zp_arr2 - true_ordered) ** 2)) * 1000

        print(f"Initial ZP RMS: {rms_initial:.3f} mmag")
        print(f"Cleaned ZP RMS: {rms_cleaned:.3f} mmag")
        print(f"Flagged {len(flagged_stars)} stars, "
              f"of which {len(flagged_stars & variable_star_ids)} are true variables")

        # Cleaning should improve the ZP RMS
        assert rms_cleaned < rms_initial, \
            f"Cleaning did not improve: {rms_cleaned:.3f} >= {rms_initial:.3f}"

        # Should catch most variable stars
        recall = len(flagged_stars & variable_star_ids) / len(variable_star_ids)
        print(f"Variable star recall: {recall:.1%}")
        assert recall > 0.5, f"Recall too low: {recall:.1%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
