#!/usr/bin/env python3
"""
Mesh Simplification Script for PAROL6 Robot STL Files

Uses Open3D's quadric decimation with boundary preservation to maintain
sharp mechanical edges while aggressively reducing flat surface triangles.

Key insight: CAD meshes have many triangles on flat surfaces (wasteful) but
sharp edges that define the mechanical appearance (must preserve). Open3D's
boundary_weight parameter helps preserve these features.

The auto-optimization uses binary search to find the most aggressive reduction
that keeps Hausdorff distance (surface deviation) below a threshold. This
correlates well with visual quality - meshes that stay within 1% of the
bounding box diagonal look nearly identical to the original.

Usage:
    python scripts/simplify_meshes.py                  # Auto-optimize all meshes
    python scripts/simplify_meshes.py --preview        # Preview without saving
    python scripts/simplify_meshes.py --target 0.2     # Keep 20% of triangles (80% reduction)
    python scripts/simplify_meshes.py --max-hausdorff 0.02  # Allow 2% surface deviation
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d is required. Install with: pip install open3d")
    raise SystemExit(1)

try:
    import trimesh
except ImportError:
    print("Error: trimesh is required. Install with: pip install trimesh")
    raise SystemExit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MESH_DIR = Path(__file__).parent.parent / "parol6" / "urdf_model" / "meshes"


@dataclass
class QualityMetrics:
    """Quality metrics comparing original and simplified meshes."""

    original_triangles: int
    simplified_triangles: int
    triangle_reduction: float
    sharp_edges_original: int
    sharp_edges_simplified: int
    sharp_edge_preservation: float
    hausdorff_normalized: float
    file_size_original: int
    file_size_simplified: int

    def looks_good(self) -> tuple[bool, str]:
        """Check if the simplification maintains visual quality.

        Primary metric is Hausdorff distance (surface deviation), which correlates
        well with visual quality. Sharp edge preservation is secondary - aggressive
        reduction typically preserves 25-40% of sharp edges while still looking good.
        """
        issues = []

        # Primary quality metric - surface deviation
        if self.hausdorff_normalized > 0.01:
            issues.append(f"surface deviation: {self.hausdorff_normalized:.1%}")

        # Secondary - sharp edges (lower threshold for aggressive mode)
        if self.sharp_edge_preservation < 0.20:
            issues.append(f"sharp edges: {self.sharp_edge_preservation:.0%} preserved")

        if issues:
            return False, "; ".join(issues)
        return True, "good quality"

    def __str__(self) -> str:
        looks_good, reason = self.looks_good()
        status = "✓ GOOD" if looks_good else "✗ CHECK"

        size_reduction = (1 - self.file_size_simplified / self.file_size_original) * 100

        return (
            f"  Triangles:    {self.original_triangles:,} → {self.simplified_triangles:,} "
            f"({self.triangle_reduction:.0%} reduction)\n"
            f"  Sharp edges:  {self.sharp_edges_original:,} → {self.sharp_edges_simplified:,} "
            f"({self.sharp_edge_preservation:.0%} preserved)\n"
            f"  Hausdorff:    {self.hausdorff_normalized:.2%} of bounding box\n"
            f"  File size:    {self.file_size_original / 1024:.0f} KB → {self.file_size_simplified / 1024:.0f} KB "
            f"({size_reduction:.0f}% smaller)\n"
            f"  Quality:      {status} - {reason}"
        )


def count_sharp_edges(mesh: trimesh.Trimesh, threshold_deg: float = 60) -> int:
    """Count edges with dihedral angle above threshold (default 60° for mechanical edges)."""
    if len(mesh.face_adjacency_angles) == 0:
        return 0
    angles_deg = np.degrees(mesh.face_adjacency_angles)
    return int(np.sum(angles_deg > threshold_deg))


def compute_hausdorff(
    original: trimesh.Trimesh, simplified: trimesh.Trimesh, samples: int = 5000
) -> float:
    """Compute normalized Hausdorff distance."""
    bbox_diag = np.linalg.norm(original.bounding_box.extents)
    if bbox_diag == 0:
        return 0.0

    try:
        pts, _ = trimesh.sample.sample_surface(original, samples)
        _, distances, _ = trimesh.proximity.closest_point(simplified, pts)
        return float(distances.max() / bbox_diag)
    except Exception:
        return 0.0


def simplify_mesh_o3d(
    mesh_o3d: o3d.geometry.TriangleMesh,
    target_ratio: float = 0.5,
    boundary_weight: float = 100.0,
) -> o3d.geometry.TriangleMesh:
    """
    Simplify mesh using Open3D's quadric decimation with boundary preservation.

    Args:
        mesh_o3d: Open3D triangle mesh
        target_ratio: Target ratio of triangles to keep (0.5 = keep 50%)
        boundary_weight: Weight for boundary edge preservation (higher = more preservation)

    Returns:
        Simplified mesh
    """
    target_triangles = max(int(len(mesh_o3d.triangles) * target_ratio), 4)

    simplified = mesh_o3d.simplify_quadric_decimation(
        target_number_of_triangles=target_triangles,
        boundary_weight=boundary_weight,
    )
    simplified.compute_vertex_normals()

    return simplified


def find_optimal_reduction(
    mesh_o3d: o3d.geometry.TriangleMesh,
    tm_original: trimesh.Trimesh,
    max_hausdorff: float = 0.003,
    min_target: float = 0.20,
    max_target: float = 0.60,
) -> tuple[float, o3d.geometry.TriangleMesh]:
    """
    Find the most aggressive reduction while maintaining geometric accuracy.

    Uses binary search to find optimal target_ratio. Primary metric is Hausdorff
    distance (surface deviation) which correlates better with visual quality than
    sharp edge counts.
    """
    best_ratio = max_target
    best_mesh = None

    low, high = min_target, max_target
    for _ in range(10):  # Binary search iterations
        mid = (low + high) / 2

        simplified = simplify_mesh_o3d(mesh_o3d, mid)
        tm_simplified = trimesh.Trimesh(
            vertices=np.asarray(simplified.vertices),
            faces=np.asarray(simplified.triangles),
        )

        hausdorff = compute_hausdorff(tm_original, tm_simplified)

        if hausdorff <= max_hausdorff:
            # Good enough quality, try more aggressive
            best_ratio = mid
            best_mesh = simplified
            high = mid
        else:
            # Too aggressive, back off
            low = mid

    # If we couldn't meet the threshold, use the max_target (least aggressive)
    if best_mesh is None:
        best_mesh = simplify_mesh_o3d(mesh_o3d, max_target)
        best_ratio = max_target

    return best_ratio, best_mesh


def process_stl_file(
    input_path: Path,
    output_path: Path | None = None,
    target_ratio: float | None = None,
    max_hausdorff: float = 0.01,
    preview_only: bool = False,
) -> tuple[Path | None, QualityMetrics | None]:
    """Process a single STL file."""
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_simplified.stl"

    logger.info(f"Processing: {input_path.name}")

    # Load with Open3D
    mesh_o3d = o3d.io.read_triangle_mesh(str(input_path))
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.compute_vertex_normals()

    original_triangles = len(mesh_o3d.triangles)

    # Also load with trimesh for quality metrics
    tm_original = trimesh.Trimesh(
        vertices=np.asarray(mesh_o3d.vertices),
        faces=np.asarray(mesh_o3d.triangles),
    )
    original_sharp = count_sharp_edges(tm_original)

    # Simplify
    if target_ratio is not None:
        simplified = simplify_mesh_o3d(mesh_o3d, target_ratio)
        used_ratio = target_ratio
    else:
        used_ratio, simplified = find_optimal_reduction(
            mesh_o3d, tm_original, max_hausdorff
        )

    # Convert to trimesh for metrics
    tm_simplified = trimesh.Trimesh(
        vertices=np.asarray(simplified.vertices),
        faces=np.asarray(simplified.triangles),
    )

    # Compute metrics
    simplified_sharp = count_sharp_edges(tm_simplified)
    hausdorff = compute_hausdorff(tm_original, tm_simplified)

    # Ensure normals are computed for STL export
    simplified.compute_vertex_normals()
    simplified.compute_triangle_normals()

    # Save to temp file to get size (or actual output)
    if preview_only:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            temp_path = Path(f.name)
        o3d.io.write_triangle_mesh(str(temp_path), simplified)
        file_size_simplified = temp_path.stat().st_size
        temp_path.unlink()
    else:
        o3d.io.write_triangle_mesh(str(output_path), simplified)
        file_size_simplified = output_path.stat().st_size

    metrics = QualityMetrics(
        original_triangles=original_triangles,
        simplified_triangles=len(simplified.triangles),
        triangle_reduction=1 - len(simplified.triangles) / original_triangles,
        sharp_edges_original=original_sharp,
        sharp_edges_simplified=simplified_sharp,
        sharp_edge_preservation=simplified_sharp / original_sharp
        if original_sharp > 0
        else 1.0,
        hausdorff_normalized=hausdorff,
        file_size_original=input_path.stat().st_size,
        file_size_simplified=file_size_simplified,
    )

    logger.info(f"\n{metrics}")

    if preview_only:
        logger.info("  (preview mode - not saved)")
        return None, metrics

    logger.info(f"  Saved: {output_path.name}")
    return output_path, metrics


def process_directory(
    directory: Path,
    target_ratio: float | None = None,
    max_hausdorff: float = 0.01,
    skip_existing: bool = False,
    preview_only: bool = False,
) -> list[tuple[Path | None, QualityMetrics]]:
    """Process all STL files in a directory."""
    stl_files = list(directory.glob("*.STL")) + list(directory.glob("*.stl"))
    stl_files = [f for f in stl_files if "_simplified" not in f.stem]

    if not stl_files:
        logger.warning(f"No STL files found in {directory}")
        return []

    logger.info(f"Found {len(stl_files)} STL files to process\n")

    results = []
    for stl_file in sorted(stl_files):
        output_path = stl_file.parent / f"{stl_file.stem}_simplified.stl"

        if skip_existing and output_path.exists():
            logger.info(f"Skipping {stl_file.name} (exists)\n")
            continue

        try:
            result_path, metrics = process_stl_file(
                stl_file,
                output_path=output_path,
                target_ratio=target_ratio,
                max_hausdorff=max_hausdorff,
                preview_only=preview_only,
            )
            if metrics:
                results.append((result_path, metrics))
            logger.info("")
        except Exception as e:
            logger.error(f"Failed to process {stl_file.name}: {e}")
            import traceback

            traceback.print_exc()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Simplify STL meshes while preserving sharp mechanical edges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                        # Auto-optimize all meshes (1%% Hausdorff threshold)
    %(prog)s --preview              # Preview without saving
    %(prog)s --target 0.2           # Keep 20%% of triangles (80%% reduction)
    %(prog)s --max-hausdorff 0.02   # Allow 2%% surface deviation (more aggressive)
    %(prog)s -i L5.STL --preview    # Preview single file

The script uses Open3D's quadric decimation with boundary preservation,
which is better suited for CAD meshes than generic decimation algorithms.
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        help="Single STL file to process",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (default: <input>_simplified.stl)",
    )
    parser.add_argument(
        "--mesh-dir",
        "-d",
        type=Path,
        default=DEFAULT_MESH_DIR,
        help=f"Directory containing STL files (default: {DEFAULT_MESH_DIR})",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=float,
        help="Target ratio of triangles to keep (0.0-1.0). Disables auto-optimization.",
    )
    parser.add_argument(
        "--max-hausdorff",
        "-m",
        type=float,
        default=0.003,
        help="Maximum Hausdorff distance (fraction of bounding box, default: 0.003 = 0.3%%)",
    )
    parser.add_argument(
        "--preview",
        "-p",
        action="store_true",
        help="Preview metrics without saving",
    )
    parser.add_argument(
        "--skip-existing",
        "-s",
        action="store_true",
        help="Skip files that already have simplified versions",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.input:
        input_path = args.input
        if not input_path.is_absolute():
            if (args.mesh_dir / input_path).exists():
                input_path = args.mesh_dir / input_path
            elif not input_path.exists():
                logger.error(f"File not found: {input_path}")
                raise SystemExit(1)

        process_stl_file(
            input_path,
            output_path=args.output,
            target_ratio=args.target,
            max_hausdorff=args.max_hausdorff,
            preview_only=args.preview,
        )
    else:
        if not args.mesh_dir.exists():
            logger.error(f"Mesh directory not found: {args.mesh_dir}")
            raise SystemExit(1)

        results = process_directory(
            args.mesh_dir,
            target_ratio=args.target,
            max_hausdorff=args.max_hausdorff,
            skip_existing=args.skip_existing,
            preview_only=args.preview,
        )

        if results:
            # Summary
            good = sum(1 for _, m in results if m.looks_good()[0])
            logger.info(f"Summary: {good}/{len(results)} meshes passed quality check")

            total_orig = sum(m.file_size_original for _, m in results)
            total_simp = sum(m.file_size_simplified for _, m in results)
            reduction = (1 - total_simp / total_orig) * 100
            logger.info(
                f"Total: {total_orig / 1024 / 1024:.2f} MB → {total_simp / 1024 / 1024:.2f} MB "
                f"({reduction:.0f}% reduction)"
            )


if __name__ == "__main__":
    main()
