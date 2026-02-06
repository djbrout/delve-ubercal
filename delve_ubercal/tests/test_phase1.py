"""Unit tests for Phase 1: Overlap Graph and Connectivity."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from delve_ubercal.phase1_overlap_graph import UnionFind, build_overlap_graph
from delve_ubercal.phase0_ingest import load_config


@pytest.fixture
def config():
    return load_config()


class TestUnionFind:
    def test_basic_union(self):
        uf = UnionFind()
        uf.union("a", "b")
        assert uf.connected("a", "b")

    def test_not_connected(self):
        uf = UnionFind()
        uf.find("a")
        uf.find("b")
        assert not uf.connected("a", "b")

    def test_transitive(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        assert uf.connected("a", "c")

    def test_components(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        uf.union("d", "e")
        uf.find("f")  # singleton
        comps = uf.components()
        assert len(comps) == 3
        # Find which component has a, b, c
        for root, members in comps.items():
            if "a" in members:
                assert set(members) == {"a", "b", "c"}
            elif "d" in members:
                assert set(members) == {"d", "e"}
            else:
                assert set(members) == {"f"}

    def test_tuple_keys(self):
        """Test with (expnum, ccdnum) tuples as keys."""
        uf = UnionFind()
        uf.union((100, 1), (100, 2))
        uf.union((200, 1), (200, 2))
        uf.union((100, 2), (200, 1))  # Bridge the two exposures
        assert uf.connected((100, 1), (200, 2))

    def test_many_nodes(self):
        """Test with many nodes for performance."""
        uf = UnionFind()
        # Chain 1000 nodes
        for i in range(999):
            uf.union(i, i + 1)
        assert uf.connected(0, 999)
        comps = uf.components()
        assert len(comps) == 1


class TestSyntheticOverlapGraph:
    """Test overlap graph with synthetic data."""

    def _make_synthetic_detections(self, tmpdir):
        """Create synthetic Phase 0 output with known connectivity.

        Creates 3 components:
        - Component 1 (DES, 5 nodes): (100,1), (200,1), (200,2), (300,1), (300,2)
          connected via shared stars
        - Component 2 (DES, 2 nodes): (400,1), (400,2) isolated
        - Component 3 (non-DES, 2 nodes): (500,1), (500,2) isolated
        """
        nside = 32
        pixel = 8657

        # Star 1: (100,1) and (200,1) — connects exp 100 and 200
        # Star 2: (200,1) and (200,2) — connects CCD 1 and 2 within exp 200
        # Star 3: (200,2) and (300,1) — connects exp 200 and 300
        # Star 4: (300,1) and (300,2) — connects CCD 1 and 2 within exp 300
        # Star 5: (100,1) and (300,2) — redundant bridge
        # -> All 5 nodes in component 1 are connected
        #
        # Star 6: (400,1) and (400,2) — isolated DES component
        # Star 7: (500,1) and (500,2) — isolated non-DES component

        detections = pd.DataFrame({
            "objectid": [
                "star1", "star1",
                "star2", "star2",
                "star3", "star3",
                "star4", "star4",
                "star5", "star5",
                "star6", "star6",
                "star7", "star7",
            ],
            "expnum": [
                100, 200,
                200, 200,
                200, 300,
                300, 300,
                100, 300,
                400, 400,
                500, 500,
            ],
            "ccdnum": [
                1, 1,
                1, 2,
                2, 1,
                1, 2,
                1, 2,
                1, 2,
                1, 2,
            ],
            "band": ["g"] * 14,
            "m_inst": [18.0] * 14,
            "m_err": [0.01] * 14,
            "mjd": [57000.0] * 14,
            "ra": [60.0] * 14,
            "dec": [-35.0] * 14,
            "x": [1000.0] * 14,
            "y": [2000.0] * 14,
        })

        # Write as Phase 0 output
        phase0_dir = tmpdir / "phase0_g"
        phase0_dir.mkdir(parents=True, exist_ok=True)
        detections.to_parquet(
            phase0_dir / f"detections_nside{nside}_pixel{pixel}.parquet",
            index=False,
        )

        # Create DES FGCM file (exposures 100-400 are DES, 500 is not)
        fgcm = pd.DataFrame({
            "expnum": [100, 200, 300, 400],
            "ccdnum": [1, 1, 1, 1],
            "band": ["g"] * 4,
            "mag_zero": [25.0] * 4,
            "sigma_mag_zero": [0.001] * 4,
            "source": ["FGCM"] * 4,
            "flag": [0] * 4,
        })
        cache_dir = tmpdir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fgcm.to_parquet(cache_dir / "des_y6_fgcm_zeropoints.parquet", index=False)

        return tmpdir, cache_dir, [pixel]

    def test_component_structure(self, config, tmp_path):
        """Verify correct component identification."""
        output_dir, cache_dir, pixels = self._make_synthetic_detections(tmp_path)

        stats = build_overlap_graph(
            "g", np.array(pixels), config, output_dir, cache_dir
        )

        # Component 1: (100,1), (200,1), (200,2), (300,1), (300,2) — 5 nodes
        # Component 2: (400,1), (400,2) — 2 nodes, DES but disconnected
        # Component 3: (500,1), (500,2) — 2 nodes, non-DES
        assert stats["n_nodes_total"] == 9
        assert stats["n_components"] == 3  # 3 components total

    def test_des_component_is_largest(self, config, tmp_path):
        """The DES-connected component should be the largest with DES exposures."""
        output_dir, cache_dir, pixels = self._make_synthetic_detections(tmp_path)

        stats = build_overlap_graph(
            "g", np.array(pixels), config, output_dir, cache_dir
        )

        # The largest DES component has 5 nodes (exp 100, 200, 300)
        # exp 400 component (2 nodes) is also DES but smaller
        # DES connected = 5 (largest DES component)
        assert stats["n_nodes_connected"] == 5

    def test_disconnected_dropped(self, config, tmp_path):
        """Non-DES-connected nodes should be dropped."""
        output_dir, cache_dir, pixels = self._make_synthetic_detections(tmp_path)

        stats = build_overlap_graph(
            "g", np.array(pixels), config, output_dir, cache_dir
        )

        # Dropped: exp 400 component (2 nodes, DES but disconnected) +
        #          exp 500 component (2 nodes, non-DES) = 4
        assert stats["n_nodes_dropped"] == 4

    def test_des_nodes_identified(self, config, tmp_path):
        """DES CCD-exposures should be identified in the connected component."""
        output_dir, cache_dir, pixels = self._make_synthetic_detections(tmp_path)

        stats = build_overlap_graph(
            "g", np.array(pixels), config, output_dir, cache_dir
        )

        # DES nodes in connected component: (100,1), (200,1), (200,2), (300,1), (300,2)
        # All 5 are DES (exps 100, 200, 300 are in FGCM)
        assert stats["n_des_nodes"] == 5

    def test_output_files_exist(self, config, tmp_path):
        """Verify output files are created."""
        output_dir, cache_dir, pixels = self._make_synthetic_detections(tmp_path)

        build_overlap_graph("g", np.array(pixels), config, output_dir, cache_dir)

        phase1_dir = output_dir / "phase1_g"
        assert (phase1_dir / "connected_nodes.parquet").exists()
        assert (phase1_dir / "dropped_nodes.parquet").exists()
        assert (phase1_dir / "des_nodes.parquet").exists()

    def test_connected_nodes_file(self, config, tmp_path):
        """Connected nodes file has correct content."""
        output_dir, cache_dir, pixels = self._make_synthetic_detections(tmp_path)

        build_overlap_graph("g", np.array(pixels), config, output_dir, cache_dir)

        connected = pd.read_parquet(output_dir / "phase1_g" / "connected_nodes.parquet")
        assert len(connected) == 5
        assert set(connected.columns) == {"expnum", "ccdnum"}

    def test_star_lists_saved(self, config, tmp_path):
        """Per-pixel star detection lists are saved for Phase 2."""
        output_dir, cache_dir, pixels = self._make_synthetic_detections(tmp_path)

        build_overlap_graph("g", np.array(pixels), config, output_dir, cache_dir)

        star_file = output_dir / "phase1_g" / f"star_lists_nside32_pixel{pixels[0]}.parquet"
        assert star_file.exists()
        star_lists = pd.read_parquet(star_file)
        assert "objectid" in star_lists.columns
        assert "expnum" in star_lists.columns
        assert "m_inst" in star_lists.columns


class TestEdgeStatistics:
    """Test edge statistics computation."""

    def test_edge_counts(self, config, tmp_path):
        """Verify shared star counts per edge are correct."""
        nside = 32
        pixel = 8657

        # 3 stars all observed on (100,1) and (200,1)
        # -> edge ((100,1),(200,1)) should have 3 shared stars
        detections = pd.DataFrame({
            "objectid": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "expnum": [100, 200, 100, 200, 100, 200],
            "ccdnum": [1, 1, 1, 1, 1, 1],
            "band": ["g"] * 6,
            "m_inst": [18.0] * 6,
            "m_err": [0.01] * 6,
            "mjd": [57000.0] * 6,
            "ra": [60.0] * 6,
            "dec": [-35.0] * 6,
            "x": [1000.0] * 6,
            "y": [2000.0] * 6,
        })

        phase0_dir = tmp_path / "phase0_g"
        phase0_dir.mkdir(parents=True, exist_ok=True)
        detections.to_parquet(
            phase0_dir / f"detections_nside{nside}_pixel{pixel}.parquet",
            index=False,
        )

        fgcm = pd.DataFrame({
            "expnum": [100, 200], "ccdnum": [1, 1],
            "band": ["g", "g"], "mag_zero": [25.0, 25.0],
            "sigma_mag_zero": [0.001, 0.001],
            "source": ["FGCM", "FGCM"], "flag": [0, 0],
        })
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fgcm.to_parquet(cache_dir / "des_y6_fgcm_zeropoints.parquet", index=False)

        stats = build_overlap_graph(
            "g", np.array([pixel]), config, tmp_path, cache_dir
        )

        assert stats["n_edges"] == 1
        assert stats["edge_stars_min"] == 3
        assert stats["edge_stars_max"] == 3
        assert stats["edge_stars_median"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
