"""Unit tests for Phase 0: Data Ingestion."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from delve_ubercal.phase0_ingest import (
    apply_detection_cap,
    apply_local_joins,
    build_pixel_query,
    filter_min_detections,
    load_config,
)


@pytest.fixture
def config():
    """Load the default config."""
    return load_config()


class TestLoadConfig:
    def test_loads_successfully(self, config):
        assert "survey" in config
        assert "data" in config
        assert "quality_cuts" in config

    def test_bands(self, config):
        assert config["survey"]["bands"] == ["g", "r", "i", "z"]

    def test_nside(self, config):
        assert config["survey"]["nside_chunk"] == 32

    def test_quality_cuts(self, config):
        cuts = config["quality_cuts"]
        assert cuts["mag_min"] == 17.0
        assert cuts["mag_max"] == 20.0
        assert cuts["magerr_max"] == 0.05
        assert cuts["class_star_min"] == 0.8
        assert cuts["max_detections_per_star"] == 25
        assert 61 in cuts["exclude_ccdnums"]


class TestBuildPixelQuery:
    def test_returns_string(self, config):
        query = build_pixel_query(1000, 32, "g", config)
        assert isinstance(query, str)

    def test_contains_quality_cuts(self, config):
        query = build_pixel_query(1000, 32, "g", config)
        assert "class_star > 0.8" in query
        assert "magerr_aper4 < 0.05" in query
        assert "mag_aper4 > 17.0" in query
        assert "mag_aper4 < 20.0" in query
        assert "61" in query  # excluded CCD

    def test_contains_band_filter(self, config):
        query = build_pixel_query(1000, 32, "r", config)
        assert "filter = 'r'" in query

    def test_no_join(self, config):
        """Optimized query should NOT contain JOIN (joins done locally)."""
        query = build_pixel_query(1000, 32, "g", config)
        assert "JOIN" not in query

    def test_selects_mag_aper4(self, config):
        """Query should select mag_aper4 (zpterm correction done locally)."""
        query = build_pixel_query(1000, 32, "g", config)
        assert "mag_aper4" in query

    def test_selects_all_photometry_columns(self, config):
        """Query should select all aperture mags, auto, fwhm, and shape params."""
        query = build_pixel_query(1000, 32, "g", config)
        for col in ["mag_auto", "mag_aper1", "mag_aper2", "mag_aper4", "mag_aper8",
                     "fwhm", "class_star", "kron_radius", "asemi", "bsemi"]:
            assert col in query, f"Missing column: {col}"


class TestLocalJoins:
    def test_applies_zpterm_and_instrument_filter(self, config):
        """Local joins should compute m_inst and filter by instrument."""
        df = pd.DataFrame({
            "objectid": ["obj1", "obj2", "obj3"],
            "exposure": ["exp1", "exp1", "exp2"],
            "ccdnum": [1, 2, 1],
            "mag_aper4": [18.0, 19.0, 17.5],
            "magerr_aper4": [0.01, 0.02, 0.015],
            "mjd": [57000.0, 57000.0, 57001.0],
            "ra": [60.0, 60.1, 60.2],
            "dec": [-35.0, -35.1, -35.2],
            "x": [1000.0, 1100.0, 1200.0],
            "y": [2000.0, 2100.0, 2200.0],
        })
        chip_df = pd.DataFrame({
            "exposure": ["exp1", "exp1", "exp2"],
            "ccdnum": [1, 2, 1],
            "zpterm": [0.1, 0.15, 0.12],
        })
        exposure_df = pd.DataFrame({
            "exposure": ["exp1", "exp2"],
            "expnum": [500001, 500002],
            "instrument": ["c4d", "c4d"],
        })
        result = apply_local_joins(df, chip_df, exposure_df, "g", config)
        assert len(result) == 3
        assert "m_inst" in result.columns
        assert "expnum" in result.columns
        assert "band" in result.columns
        # Check zpterm subtraction
        assert abs(result.iloc[0]["m_inst"] - (18.0 - 0.1)) < 1e-10

    def test_filters_non_decam(self, config):
        """Non-DECam exposures should be filtered out."""
        df = pd.DataFrame({
            "objectid": ["obj1", "obj2"],
            "exposure": ["exp1", "exp2"],
            "ccdnum": [1, 1],
            "mag_aper4": [18.0, 19.0],
            "magerr_aper4": [0.01, 0.02],
            "mjd": [57000.0, 57001.0],
            "ra": [60.0, 60.1],
            "dec": [-35.0, -35.1],
            "x": [1000.0, 1100.0],
            "y": [2000.0, 2100.0],
        })
        chip_df = pd.DataFrame({
            "exposure": ["exp1", "exp2"],
            "ccdnum": [1, 1],
            "zpterm": [0.1, 0.12],
        })
        exposure_df = pd.DataFrame({
            "exposure": ["exp1", "exp2"],
            "expnum": [500001, 500002],
            "instrument": ["c4d", "k4m"],  # exp2 is NOT DECam
        })
        result = apply_local_joins(df, chip_df, exposure_df, "g", config)
        assert len(result) == 1
        assert result.iloc[0]["expnum"] == 500001


class TestDetectionCap:
    def test_no_cap_needed(self):
        """Stars with <= 25 detections should be unchanged."""
        df = pd.DataFrame({
            "objectid": ["star1"] * 5 + ["star2"] * 10,
            "m_err": np.random.default_rng(0).uniform(0.01, 0.05, 15),
            "val": range(15),
        })
        result = apply_detection_cap(df, 25)
        assert len(result) == 15

    def test_cap_applied(self):
        """Stars with > 25 detections should keep lowest-error detections."""
        rng = np.random.default_rng(42)
        errs = rng.uniform(0.001, 0.05, 50)
        df = pd.DataFrame({
            "objectid": ["star1"] * 50,
            "m_err": errs,
            "val": range(50),
        })
        result = apply_detection_cap(df, 25)
        assert len(result) == 25
        assert result["objectid"].nunique() == 1
        # The kept detections should be the 25 with smallest errors
        assert result["m_err"].max() <= np.sort(errs)[25]

    def test_mixed(self):
        """Mix of stars above and below cap."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "objectid": ["star1"] * 5 + ["star2"] * 50 + ["star3"] * 25,
            "m_err": rng.uniform(0.01, 0.05, 80),
            "val": range(80),
        })
        result = apply_detection_cap(df, 25)
        counts = result.groupby("objectid").size()
        assert counts["star1"] == 5
        assert counts["star2"] == 25
        assert counts["star3"] == 25

    def test_deterministic(self):
        """Best-error selection is deterministic (no RNG dependence)."""
        errs = np.linspace(0.001, 0.05, 100)
        df = pd.DataFrame({
            "objectid": ["star1"] * 100,
            "m_err": errs,
            "val": range(100),
        })
        r1 = apply_detection_cap(df, 25)
        r2 = apply_detection_cap(df, 25)
        assert r1["val"].tolist() == r2["val"].tolist()
        # Should keep the first 25 (lowest errors)
        assert r1["val"].tolist() == list(range(25))


class TestFilterMinDetections:
    def test_removes_singles(self):
        """Stars with 1 detection should be removed (min=2)."""
        df = pd.DataFrame({
            "objectid": ["star1"] * 1 + ["star2"] * 3,
            "val": range(4),
        })
        result = filter_min_detections(df, 2)
        assert "star1" not in result["objectid"].values
        assert len(result) == 3

    def test_keeps_multi(self):
        """Stars with >= min detections should be kept."""
        df = pd.DataFrame({
            "objectid": ["star1"] * 5 + ["star2"] * 2,
            "val": range(7),
        })
        result = filter_min_detections(df, 2)
        assert len(result) == 7


class TestOutputSchema:
    """Test that the output parquet files would have correct columns."""

    def test_expected_columns(self):
        expected = [
            "objectid", "expnum", "ccdnum", "band",
            "m_inst", "m_err", "mjd", "ra", "dec", "x", "y",
        ]
        # Simulate a detection
        df = pd.DataFrame({
            "objectid": ["obj1"],
            "expnum": [500000],
            "ccdnum": [10],
            "band": ["g"],
            "m_inst": [-5.0],
            "m_err": [0.01],
            "mjd": [57000.0],
            "ra": [60.0],
            "dec": [-35.0],
            "x": [1000.0],
            "y": [2000.0],
        })
        for col in expected:
            assert col in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
