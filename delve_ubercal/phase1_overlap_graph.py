"""Phase 1: Build overlap graph and check connectivity.

Streams through Phase 0 detections by HEALPix chunk, builds a union-find
structure over (expnum, ccdnum) nodes connected by shared stars, identifies
the DES-connected component, and drops disconnected nodes.
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from delve_ubercal.phase0_ingest import get_test_patch_pixels, get_test_region_pixels, load_config
from delve_ubercal.utils.healpix_utils import get_all_healpix_pixels


# ---------------------------------------------------------------------------
# Union-Find (Disjoint Set Union) data structure
# ---------------------------------------------------------------------------

class UnionFind:
    """Weighted union-find with path compression.

    Nodes are arbitrary hashable keys, added lazily on first use.
    """

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        # Path compression
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # Union by rank
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def connected(self, x, y):
        return self.find(x) == self.find(y)

    def components(self):
        """Return dict mapping root -> list of members."""
        comp = defaultdict(list)
        for x in self.parent:
            comp[self.find(x)].append(x)
        return dict(comp)

    def __len__(self):
        return len(self.parent)


# ---------------------------------------------------------------------------
# DES exposure identification
# ---------------------------------------------------------------------------

def load_des_expnums(config, cache_dir, band):
    """Load DES exposure numbers from the cached FGCM table.

    An exposure is "DES" if it appears in the DES FGCM zero-point table.
    """
    fgcm_file = cache_dir / "des_y6_fgcm_zeropoints.parquet"
    if not fgcm_file.exists():
        raise FileNotFoundError(
            f"DES FGCM file not found: {fgcm_file}. Run Phase 0 first."
        )
    fgcm = pd.read_parquet(fgcm_file, columns=["expnum", "band"])
    des_expnums = set(fgcm[fgcm["band"] == band]["expnum"].unique())
    return des_expnums


# ---------------------------------------------------------------------------
# Main overlap graph builder
# ---------------------------------------------------------------------------

def build_overlap_graph(band, pixels, config, output_dir, cache_dir):
    """Build the overlap graph for one band.

    Streams through Phase 0 output by HEALPix pixel, collecting
    per-star detection lists and building a union-find over
    (expnum, ccdnum) nodes.

    Parameters
    ----------
    band : str
        Filter band.
    pixels : np.ndarray
        HEALPix pixel indices to process.
    config : dict
        Pipeline configuration.
    output_dir : Path
        Phase 0 output directory.
    cache_dir : Path
        Cache directory.

    Returns
    -------
    stats : dict
        Summary statistics.
    """
    nside = config["survey"]["nside_chunk"]
    phase0_dir = output_dir / f"phase0_{band}"

    # Load DES exposure set
    des_expnums = load_des_expnums(config, cache_dir, band)
    print(f"  DES FGCM exposures ({band}): {len(des_expnums):,}", flush=True)

    # Union-find over (expnum, ccdnum) nodes
    uf = UnionFind()

    # Track all nodes and edge counts
    all_nodes = set()
    # Count shared stars per pair for statistics
    edge_star_counts = defaultdict(int)

    # Per-star detection lists — collect for Phase 2
    # We'll write these out per-pixel for memory efficiency
    phase1_output_dir = output_dir / f"phase1_{band}"
    phase1_output_dir.mkdir(parents=True, exist_ok=True)

    n_pixels = len(pixels)
    n_stars_total = 0
    n_dets_total = 0
    t0 = time.time()

    for i, pixel in enumerate(pixels):
        det_file = phase0_dir / f"detections_nside{nside}_pixel{pixel}.parquet"
        if not det_file.exists():
            continue

        df = pd.read_parquet(det_file)
        if len(df) == 0:
            continue

        n_dets_total += len(df)

        # Group by objectid to get per-star detection lists
        grouped = df.groupby("objectid")

        for star_id, group in grouped:
            if len(group) < 2:
                continue

            n_stars_total += 1

            # Get unique (expnum, ccdnum) pairs for this star
            nodes = list(
                set(zip(group["expnum"].values, group["ccdnum"].values))
            )

            if len(nodes) < 2:
                continue

            # Register all nodes
            for node in nodes:
                all_nodes.add(node)
                uf.find(node)  # Ensure node exists in UF

            # Union all pairs — connect this star's CCD-exposures
            first = nodes[0]
            for node in nodes[1:]:
                uf.union(first, node)

            # Count shared stars per pair (for edge statistics)
            # Only count unique pairs (i < j)
            for a_idx in range(len(nodes)):
                for b_idx in range(a_idx + 1, len(nodes)):
                    pair = (
                        min(nodes[a_idx], nodes[b_idx]),
                        max(nodes[a_idx], nodes[b_idx]),
                    )
                    edge_star_counts[pair] += 1

        # Write per-star detection lists for this pixel (for Phase 2)
        # Only keep objectid, expnum, ccdnum, m_inst, m_err
        star_lists = df[["objectid", "expnum", "ccdnum", "m_inst", "m_err"]].copy()
        star_list_file = phase1_output_dir / f"star_lists_nside{nside}_pixel{pixel}.parquet"
        star_lists.to_parquet(star_list_file, index=False)

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (n_pixels - i - 1) / rate if rate > 0 else 0
        print(
            f"  [{i+1}/{n_pixels}] Pixel {pixel}: "
            f"{len(grouped)} stars, {len(df)} dets, "
            f"ETA {eta:.0f}s",
            flush=True,
        )

    # -----------------------------------------------------------------------
    # Identify connected components
    # -----------------------------------------------------------------------
    print(f"\n  Building connected components...", flush=True)
    components = uf.components()
    n_components = len(components)

    # Find which component contains DES exposures
    des_component_root = None
    des_nodes_found = set()

    for root, members in components.items():
        member_expnums = {node[0] for node in members}
        des_overlap = member_expnums & des_expnums
        if des_overlap:
            if des_component_root is None:
                des_component_root = root
                des_nodes_found = set(members)
            else:
                # Multiple components have DES — pick the largest
                if len(members) > len(des_nodes_found):
                    des_component_root = root
                    des_nodes_found = set(members)

    if des_component_root is None:
        print("  WARNING: No DES exposures found in any component!", flush=True)
        connected_nodes = all_nodes
        dropped_nodes = set()
    else:
        connected_nodes = des_nodes_found
        dropped_nodes = all_nodes - connected_nodes

    n_total = len(all_nodes)
    n_connected = len(connected_nodes)
    n_dropped = len(dropped_nodes)
    pct_connected = 100.0 * n_connected / n_total if n_total > 0 else 0

    # Count DES nodes in connected component
    des_in_connected = {
        node for node in connected_nodes if node[0] in des_expnums
    }

    # Component size distribution
    comp_sizes = sorted([len(m) for m in components.values()], reverse=True)

    # Edge statistics
    if edge_star_counts:
        star_counts = np.array(list(edge_star_counts.values()))
        edge_min = int(star_counts.min())
        edge_median = int(np.median(star_counts))
        edge_max = int(star_counts.max())
        edge_mean = float(star_counts.mean())
    else:
        edge_min = edge_median = edge_max = 0
        edge_mean = 0.0

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    # Save connectivity mask: set of (expnum, ccdnum) in connected component
    connected_df = pd.DataFrame(
        list(connected_nodes), columns=["expnum", "ccdnum"]
    )
    connected_df["expnum"] = connected_df["expnum"].astype(np.int64)
    connected_df["ccdnum"] = connected_df["ccdnum"].astype(np.int32)
    connected_file = phase1_output_dir / "connected_nodes.parquet"
    connected_df.to_parquet(connected_file, index=False)

    # Save dropped nodes
    if dropped_nodes:
        dropped_df = pd.DataFrame(
            list(dropped_nodes), columns=["expnum", "ccdnum"]
        )
        dropped_df["expnum"] = dropped_df["expnum"].astype(np.int64)
        dropped_df["ccdnum"] = dropped_df["ccdnum"].astype(np.int32)
    else:
        dropped_df = pd.DataFrame(columns=["expnum", "ccdnum"])
    dropped_file = phase1_output_dir / "dropped_nodes.parquet"
    dropped_df.to_parquet(dropped_file, index=False)

    # Save DES nodes
    des_nodes_df = pd.DataFrame(
        list(des_in_connected), columns=["expnum", "ccdnum"]
    )
    if len(des_nodes_df) > 0:
        des_nodes_df["expnum"] = des_nodes_df["expnum"].astype(np.int64)
        des_nodes_df["ccdnum"] = des_nodes_df["ccdnum"].astype(np.int32)
    des_file = phase1_output_dir / "des_nodes.parquet"
    des_nodes_df.to_parquet(des_file, index=False)

    total_time = time.time() - t0

    # -----------------------------------------------------------------------
    # Print report
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}", flush=True)
    print(f"Phase 1 Connectivity Report — {band}-band", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Total CCD-exposures:      {n_total:,}", flush=True)
    print(f"  Connected (DES component): {n_connected:,} ({pct_connected:.1f}%)", flush=True)
    print(f"  Dropped:                   {n_dropped:,}", flush=True)
    print(f"  DES CCD-exposures:         {len(des_in_connected):,}", flush=True)
    print(f"  Components:                {n_components}", flush=True)
    print(f"  Largest component:         {comp_sizes[0]:,}" if comp_sizes else "", flush=True)
    if len(comp_sizes) > 1:
        print(f"  Other components:          {comp_sizes[1:]}", flush=True)
    print(f"  Stars with >= 2 dets:      {n_stars_total:,}", flush=True)
    print(f"  Total detections:          {n_dets_total:,}", flush=True)
    print(f"  Unique edges:              {len(edge_star_counts):,}", flush=True)
    print(f"  Shared stars per edge:     min={edge_min}, median={edge_median}, "
          f"mean={edge_mean:.1f}, max={edge_max}", flush=True)
    print(f"  Time:                      {total_time:.1f}s", flush=True)
    print(f"{'=' * 60}", flush=True)

    stats = {
        "band": band,
        "n_nodes_total": n_total,
        "n_nodes_connected": n_connected,
        "n_nodes_dropped": n_dropped,
        "n_des_nodes": len(des_in_connected),
        "n_components": n_components,
        "pct_connected": pct_connected,
        "n_stars": n_stars_total,
        "n_detections": n_dets_total,
        "n_edges": len(edge_star_counts),
        "edge_stars_min": edge_min,
        "edge_stars_median": edge_median,
        "edge_stars_mean": edge_mean,
        "edge_stars_max": edge_max,
        "component_sizes": comp_sizes,
        "total_time_s": total_time,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Build overlap graph and check connectivity"
    )
    parser.add_argument("--band", required=True, help="Filter band (g, r, i, z)")
    parser.add_argument(
        "--test-region", action="store_true",
        help="Limit to test region RA=50-70, Dec=-40 to -25",
    )
    parser.add_argument(
        "--test-patch", action="store_true",
        help="Limit to 10x10 deg test patch RA=50-60, Dec=-35 to -25",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--max-pixels", type=int, default=None,
        help="Limit to first N pixels (for quick testing)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    nside = config["survey"]["nside_chunk"]

    output_dir = Path(config["data"]["output_path"])
    cache_dir = Path(config["data"]["cache_path"])

    # Determine which pixels to process
    if args.test_patch:
        pixels = get_test_patch_pixels(nside)
        print(f"Test patch (10x10 deg): {len(pixels)} HEALPix pixels (nside={nside})", flush=True)
    elif args.test_region:
        pixels = get_test_region_pixels(nside)
        print(f"Test region: {len(pixels)} HEALPix pixels (nside={nside})", flush=True)
    else:
        pixels = get_all_healpix_pixels(nside)
        print(f"Full sky: {len(pixels)} HEALPix pixels (nside={nside})", flush=True)

    if args.max_pixels is not None:
        pixels = pixels[: args.max_pixels]
        print(f"Limited to first {args.max_pixels} pixels", flush=True)

    print(f"Band: {args.band}", flush=True)
    print(flush=True)

    stats = build_overlap_graph(
        args.band, pixels, config, output_dir, cache_dir
    )

    return stats


if __name__ == "__main__":
    main()
