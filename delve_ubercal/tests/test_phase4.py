"""Unit tests for Phase 4: Star Flat Refinement."""

import numpy as np
import pytest

from delve_ubercal.phase0_ingest import load_config
from delve_ubercal.phase4_starflat import (
    evaluate_chebyshev_2d,
    fit_chebyshev_2d,
    get_epoch,
)


@pytest.fixture
def config():
    return load_config()


class TestEpochAssignment:
    """Test instrumental epoch assignment."""

    def test_before_all_boundaries(self, config):
        """MJD before all boundaries -> epoch 0."""
        epoch = get_epoch(56000.0, 1, config)
        assert epoch == 0

    def test_after_first_global(self, config):
        """MJD after first global boundary (56404) -> epoch 1."""
        epoch = get_epoch(56500.0, 1, config)
        # 56500 > 56404 (g-band baffling) but < 56516 (rizY) -> epoch 1
        assert epoch == 1

    def test_after_all_globals(self, config):
        """MJD after all global boundaries."""
        epoch = get_epoch(57000.0, 1, config)
        # 57000 > 56404, 56516, 56730 -> epoch 3
        assert epoch == 3

    def test_per_ccd_boundary(self, config):
        """CCD 2 has per-CCD boundaries at 56626 and 57751."""
        # Before CCD 2 failure
        e1 = get_epoch(56600.0, 2, config)
        # After CCD 2 failure, before recovery
        e2 = get_epoch(57000.0, 2, config)
        # After CCD 2 recovery
        e3 = get_epoch(58000.0, 2, config)
        # These should be different epochs
        assert e1 < e2 < e3

    def test_normal_ccd_no_per_ccd(self, config):
        """Normal CCD (e.g., CCD 10) uses only global boundaries."""
        epoch_normal = get_epoch(57000.0, 10, config)
        epoch_special = get_epoch(57000.0, 2, config)
        # CCD 2 has more boundaries so higher epoch count
        assert epoch_special >= epoch_normal


class TestChebyshevFit:
    """Test 2D Chebyshev polynomial fitting."""

    def test_constant_pattern(self):
        """Constant offset should be recovered."""
        rng = np.random.default_rng(42)
        x = rng.uniform(1, 2048, 200)
        y = rng.uniform(1, 4096, 200)
        values = np.full(200, 0.01)  # 10 mmag constant

        coeffs = fit_chebyshev_2d(x, y, values, order=3)
        recovered = evaluate_chebyshev_2d(x, y, coeffs)
        rms = np.sqrt(np.mean((recovered - values) ** 2))
        assert rms < 1e-10

    def test_linear_gradient(self):
        """Linear gradient across CCD should be recovered."""
        rng = np.random.default_rng(42)
        x = rng.uniform(1, 2048, 500)
        y = rng.uniform(1, 4096, 500)
        # Linear gradient: 10 mmag across x
        x_norm = (x - 1) / 2047.0
        values = 0.010 * x_norm  # 0 to 10 mmag

        coeffs = fit_chebyshev_2d(x, y, values, order=3)
        recovered = evaluate_chebyshev_2d(x, y, coeffs)
        rms = np.sqrt(np.mean((recovered - values) ** 2))
        assert rms < 0.001  # < 1 mmag residual

    def test_quadratic_pattern(self):
        """Quadratic bowl pattern should be recovered."""
        rng = np.random.default_rng(42)
        x = rng.uniform(1, 2048, 500)
        y = rng.uniform(1, 4096, 500)
        # Quadratic: bowl pattern
        x_norm = 2 * (x - 1) / 2047 - 1  # [-1, 1]
        y_norm = 2 * (y - 1) / 4095 - 1  # [-1, 1]
        values = 0.005 * (x_norm ** 2 + y_norm ** 2)

        coeffs = fit_chebyshev_2d(x, y, values, order=3)
        recovered = evaluate_chebyshev_2d(x, y, coeffs)
        rms = np.sqrt(np.mean((recovered - values) ** 2))
        assert rms < 0.001

    def test_noisy_recovery(self):
        """Pattern recovery with noise should still work reasonably."""
        rng = np.random.default_rng(42)
        n_pts = 1000
        x = rng.uniform(1, 2048, n_pts)
        y = rng.uniform(1, 4096, n_pts)
        x_norm = 2 * (x - 1) / 2047 - 1
        true_pattern = 0.010 * x_norm  # 10 mmag gradient
        noise = rng.normal(0, 0.020, n_pts)  # 20 mmag noise
        values = true_pattern + noise

        coeffs = fit_chebyshev_2d(x, y, values, order=3)
        recovered = evaluate_chebyshev_2d(x, y, coeffs)

        # Pattern should be roughly recovered (not perfectly due to noise)
        pattern_residual = np.sqrt(np.mean((recovered - true_pattern) ** 2))
        assert pattern_residual < 0.005  # < 5 mmag

    def test_roundtrip(self):
        """fit then evaluate should recover the input."""
        rng = np.random.default_rng(42)
        x = rng.uniform(1, 2048, 300)
        y = rng.uniform(1, 4096, 300)
        x_norm = 2 * (x - 1) / 2047 - 1
        y_norm = 2 * (y - 1) / 4095 - 1
        # T0(x)*T1(y) = 1 * y_norm = y_norm
        values = 0.005 * y_norm

        coeffs = fit_chebyshev_2d(x, y, values, order=3)
        recovered = evaluate_chebyshev_2d(x, y, coeffs)
        rms = np.sqrt(np.mean((recovered - values) ** 2))
        assert rms < 1e-10


class TestAmplitudePlausibility:
    """Test that corrections are in the expected range."""

    def test_typical_decam_amplitude(self):
        """DECam star flat corrections are typically 1-10 mmag RMS."""
        rng = np.random.default_rng(42)
        x = rng.uniform(1, 2048, 500)
        y = rng.uniform(1, 4096, 500)
        # Typical 5 mmag amplitude pattern
        x_norm = 2 * (x - 1) / 2047 - 1
        values = 0.005 * x_norm  # 5 mmag gradient

        coeffs = fit_chebyshev_2d(x, y, values, order=3)
        recovered = evaluate_chebyshev_2d(x, y, coeffs)
        rms = np.sqrt(np.mean(recovered ** 2)) * 1000

        # RMS of the correction should be a few mmag
        assert 1 < rms < 10, f"Correction RMS {rms:.1f} mmag outside expected range"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
