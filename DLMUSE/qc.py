"""
Quality control for DLMUSE segmentations.

Checks left/right hemisphere ROI integrity: every lateralized MUSE ROI should
lie in its labelled hemisphere, and each left ROI should have a right
counterpart (and vice versa).

The midsagittal plane is fitted (least squares) through the midpoints of the
left/right ROI centroids, so it follows head tilt/rotation. Its normal is
oriented towards world +x (Right in nibabel's RAS+ space, taken from the NIfTI
affine), so the check works for any voxel orientation (e.g. LPS) and a global
L/R label swap is still detected: right ROIs should lie on the positive side of
the plane, left ROIs on the negative side.
"""

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np

DEFAULT_MAPPING_CSV = (
    Path(__file__).parent / "shared" / "dicts" / "MUSE_mapping_consecutive_indices.csv"
)
QC_CSV_NAME = "qc_report.csv"

# A WARN is raised when more than this fraction of an ROI's voxels lies in
# the opposite hemisphere, by more than MIDLINE_TOLERANCE_MM. The tolerance
# absorbs structures that legitimately abut the midline (fornix, medial cortex).
MAX_WRONG_SIDE_FRACTION = 0.10
MIDLINE_TOLERANCE_MM = 2.0
# A WARN is raised when |L - R| / ((L + R) / 2) exceeds this value
# (1.0 means one side is 3x the volume of the other).
MAX_ASYMMETRY_INDEX = 1.0
# Minimum number of bilateral pairs needed to fit the midsagittal plane;
# below this, a plane x = mean(x) is used instead.
MIN_PAIRS_FOR_PLANE_FIT = 10



@dataclass(frozen=True)
class QCThresholds:
    """Tunable QC thresholds; defaults are the module-level constants above."""

    max_wrong_side_fraction: float = MAX_WRONG_SIDE_FRACTION
    midline_tolerance_mm: float = MIDLINE_TOLERANCE_MM
    max_asymmetry_index: float = MAX_ASYMMETRY_INDEX
    min_pairs_for_plane_fit: int = MIN_PAIRS_FOR_PLANE_FIT
    # Also write the per-ROI detailed CSV (not a threshold, but set the same way)
    detail: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_wrong_side_fraction <= 1.0:
            raise ValueError(
                f"max_wrong_side_fraction must be in [0, 1]. Got: {self.max_wrong_side_fraction}."
            )
        if self.midline_tolerance_mm < 0:
            raise ValueError(
                f"midline_tolerance_mm must be >= 0. Got: {self.midline_tolerance_mm}."
            )
        if self.max_asymmetry_index < 0:
            raise ValueError(
                f"max_asymmetry_index must be >= 0. Got: {self.max_asymmetry_index}."
            )
        # A plane needs at least 3 non-collinear points
        if self.min_pairs_for_plane_fit < 3:
            raise ValueError(
                f"min_pairs_for_plane_fit must be >= 3. Got: {self.min_pairs_for_plane_fit}."
            )


# Keys accepted by parse_qc_thresholds (short CLI names and the full field names)
QC_THRESHOLD_KEYS = {
    "max_wrong_side": "max_wrong_side_fraction",
    "midline_tolerance": "midline_tolerance_mm",
    "max_asymmetry": "max_asymmetry_index",
    "min_pairs": "min_pairs_for_plane_fit",
    "QC_detail": "detail",
}
QC_THRESHOLD_KEYS.update({v: v for v in list(QC_THRESHOLD_KEYS.values())})


def _parse_threshold_string(spec: str) -> dict:
    """Parse a JSON object; fall back to `{key: value, ...}` with unquoted keys
    (e.g. when the shell has stripped the inner double quotes)."""
    spec = spec.strip()
    try:
        parsed = json.loads(spec)
    except json.JSONDecodeError:
        body = spec[1:-1] if spec.startswith("{") and spec.endswith("}") else None
        if body is None:
            raise ValueError(
                f"QC thresholds must be a JSON object such as "
                f'\'{{"max_wrong_side": 0.2}}\'. Got: {spec}'
            ) from None
        parsed = {}
        for item in filter(None, (s.strip() for s in body.split(","))):
            key, sep, value = item.partition(":")
            if not sep:
                raise ValueError(f"Invalid QC threshold entry '{item}' in: {spec}")
            parsed[key.strip().strip("'\"")] = value.strip().strip("'\"")
    if not isinstance(parsed, dict):
        raise ValueError(f"QC thresholds must be a JSON object. Got: {spec}")
    return parsed


def parse_qc_thresholds(
    spec: Union[None, str, Dict[str, Any], "QCThresholds"],
) -> "QCThresholds":
    """
    Build QCThresholds from a (partial) dict or JSON string of overrides, e.g.
    '{"max_wrong_side": 0.2, "midline_tolerance": 3}'. Keys that are not given
    keep their defaults. Accepted keys: max_wrong_side, midline_tolerance,
    max_asymmetry, min_pairs (or the full QCThresholds field names), and
    QC_detail (1 = also write the per-ROI detailed CSV; default 0).
    """
    if spec is None:
        return QCThresholds()
    if isinstance(spec, QCThresholds):
        return spec
    overrides = _parse_threshold_string(spec) if isinstance(spec, str) else dict(spec)

    unknown = sorted(set(overrides) - set(QC_THRESHOLD_KEYS))
    if unknown:
        valid = ", ".join(k for k in QC_THRESHOLD_KEYS if k != QC_THRESHOLD_KEYS[k])
        raise ValueError(f"Unknown QC threshold key(s): {unknown}. Valid keys: {valid}.")

    kwargs: Dict[str, Any] = {}
    for key, value in overrides.items():
        field = QC_THRESHOLD_KEYS[key]
        if field in kwargs:
            raise ValueError(f"QC threshold '{field}' was given more than once.")
        if field == "detail":
            if isinstance(value, str):
                value = value.strip().lower()
            if value in (1, "1", True, "true"):
                kwargs[field] = True
            elif value in (0, "0", False, "false"):
                kwargs[field] = False
            else:
                raise ValueError(f"QC option '{key}' must be 0 or 1. Got: {value!r}")
            continue
        cast = int if field == "min_pairs_for_plane_fit" else float
        try:
            if cast is int and float(value) != int(float(value)):
                raise ValueError
            kwargs[field] = cast(float(value))
        except (TypeError, ValueError):
            raise ValueError(
                f"QC threshold '{key}' must be {'an integer' if cast is int else 'a number'}. "
                f"Got: {value!r}"
            ) from None
    return QCThresholds(**kwargs)


PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_STATUS_RANK = {PASS: 0, WARN: 1, FAIL: 2}

_SIDE_RE = re.compile(r"(^(left|right)_)|(_(left|right)$)")

QC_SUMMARY_COLUMNS = ["file", "status", "issues"]

QC_COLUMNS = [
    "file",
    "roi",
    "left_index",
    "right_index",
    "left_name",
    "right_name",
    "left_voxels",
    "right_voxels",
    "left_volume_mm3",
    "right_volume_mm3",
    "asymmetry_index",
    "left_centroid_distance_mm",
    "right_centroid_distance_mm",
    "left_wrong_side_fraction",
    "right_wrong_side_fraction",
    "status",
    "issues",
]


@dataclass(frozen=True)
class ROIPair:
    roi: str
    left_index: int
    right_index: int
    left_name: str
    right_name: str


def load_roi_pairs(mapping_csv: Path = DEFAULT_MAPPING_CSV) -> List[ROIPair]:
    """
    Build the list of left/right ROI pairs from the MUSE mapping CSV.

    Labels are matched on IndexConsecutive, which is the label space DLMUSE
    writes. ROIs whose names carry no side token (e.g. brain_stem) are skipped.
    """
    sides: Dict[str, Dict[str, tuple]] = {}
    with open(mapping_csv, newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            name = row["ROINameMUSE"].strip()
            match = _SIDE_RE.search(name)
            if match is None:
                continue
            side = match.group(2) or match.group(4)
            base = _SIDE_RE.sub("", name)
            sides.setdefault(base, {})[side] = (int(row["IndexConsecutive"]), name)

    pairs = []
    for base, entry in sides.items():
        if "left" not in entry or "right" not in entry:
            continue  # no contralateral counterpart in the mapping; cannot be checked
        (l_idx, l_name), (r_idx, r_name) = entry["left"], entry["right"]
        pairs.append(ROIPair(base, l_idx, r_idx, l_name, r_name))
    return pairs


def load_valid_labels(mapping_csv: Path = DEFAULT_MAPPING_CSV) -> set:
    with open(mapping_csv, newline="", encoding="utf-8") as fp:
        return {int(row["IndexConsecutive"]) for row in csv.DictReader(fp)}


def _worse(a: str, b: str) -> str:
    return a if _STATUS_RANK[a] >= _STATUS_RANK[b] else b


def qc_mask(
    mask_path: str,
    pairs: List[ROIPair],
    valid_labels: Optional[set] = None,
    thresholds: Optional[QCThresholds] = None,
) -> Tuple[List[dict], List[str]]:
    """
    Run the L/R integrity check on one segmentation.

    Returns (one row per ROI pair, file-level issues that concern the whole
    segmentation rather than a single ROI pair).
    """
    th = thresholds or QCThresholds()
    img = nib.load(mask_path)
    labels = np.asanyarray(img.dataobj).astype(np.int64, copy=False)
    fname = os.path.basename(mask_path)
    file_issues: List[str] = []

    fg = np.nonzero(labels)
    if fg[0].size == 0:
        rows = [_row(fname, p, status=FAIL, issues="empty segmentation") for p in pairs]
        return rows, ["empty segmentation"]

    if valid_labels is not None:
        unexpected = sorted(set(np.unique(labels[fg]).tolist()) - valid_labels)
        if unexpected:
            file_issues.append(f"labels not in MUSE mapping: {unexpected}")

    fg_labels = labels[fg]
    affine = img.affine
    # World (RAS+) coordinates of every foreground voxel, shape (N, 3)
    world = (affine[:3, :3] @ np.vstack(fg).astype(np.float64)).T + affine[:3, 3]
    voxel_vol = float(abs(np.linalg.det(affine[:3, :3])))

    n_bins = max(int(fg_labels.max()), max(max(p.left_index, p.right_index) for p in pairs)) + 1
    counts = np.bincount(fg_labels, minlength=n_bins)
    centroids = np.stack(
        [np.bincount(fg_labels, weights=world[:, d], minlength=n_bins) for d in range(3)],
        axis=1,
    ) / np.maximum(counts, 1)[:, None]

    normal, origin, n_mids = _fit_midsagittal_plane(
        pairs, counts, centroids, world, th.min_pairs_for_plane_fit
    )
    if n_mids < th.min_pairs_for_plane_fit:
        file_issues.append(
            f"only {n_mids} bilateral pairs present; using x = mean(x) as midline"
        )
    elif normal[0] < 0.9:
        file_issues.append(
            f"midsagittal plane is tilted "
            f"{np.degrees(np.arccos(np.clip(normal[0], -1, 1))):.1f} deg from the world x-axis"
        )
    dist = (world - origin) @ normal  # signed distance, > 0 => right hemisphere
    centroid_dist = (centroids - origin) @ normal
    tol = th.midline_tolerance_mm
    n_far_right = np.bincount(fg_labels[dist > tol], minlength=n_bins)
    n_far_left = np.bincount(fg_labels[dist < -tol], minlength=n_bins)

    def stats(idx: int, wrong_side_counts: np.ndarray):
        n = int(counts[idx])
        if n == 0:
            return 0, None, None
        return n, float(centroid_dist[idx]), int(wrong_side_counts[idx]) / n

    rows = []
    for p in pairs:
        l_n, l_cx, l_wrong = stats(p.left_index, n_far_right)
        r_n, r_cx, r_wrong = stats(p.right_index, n_far_left)
        status, issues = PASS, []

        asym = abs(l_n - r_n) / ((l_n + r_n) / 2) if (l_n and r_n) else None

        if not l_n and not r_n:
            # Not a hemispheric integrity problem (some labels are never predicted)
            issues.append("absent in both hemispheres")
        elif not l_n or not r_n:
            status = _worse(status, FAIL)
            issues.append(f"missing {'left' if not l_n else 'right'} ROI")

        if l_n and l_cx >= 0:
            status = _worse(status, FAIL)
            issues.append("left ROI centroid in right hemisphere")
        if r_n and r_cx <= 0:
            status = _worse(status, FAIL)
            issues.append("right ROI centroid in left hemisphere")

        for side, frac in (("left", l_wrong), ("right", r_wrong)):
            if frac is not None and frac > th.max_wrong_side_fraction:
                status = _worse(status, WARN)
                issues.append(f"{frac:.1%} of {side} ROI crosses midline")

        if asym is not None and asym > th.max_asymmetry_index:
            status = _worse(status, WARN)
            issues.append(f"high L/R volume asymmetry ({asym:.2f})")

        rows.append(
            _row(
                fname,
                p,
                left_voxels=l_n,
                right_voxels=r_n,
                left_volume_mm3=round(l_n * voxel_vol, 3),
                right_volume_mm3=round(r_n * voxel_vol, 3),
                asymmetry_index=_round(asym),
                left_centroid_distance_mm=_round(l_cx, 2),
                right_centroid_distance_mm=_round(r_cx, 2),
                left_wrong_side_fraction=_round(l_wrong),
                right_wrong_side_fraction=_round(r_wrong),
                status=status,
                issues="; ".join(issues),
            )
        )
    return rows, file_issues


def _fit_midsagittal_plane(pairs, counts, centroids, world, min_pairs):
    """
    Fit the midsagittal plane through the midpoints of bilateral ROI centroids.

    Returns (unit normal pointing to world +x, point on plane, number of
    bilateral pairs used). With fewer than min_pairs pairs, the plane
    x = mean(x) is returned instead.
    """
    mids = np.array(
        [
            (centroids[p.left_index] + centroids[p.right_index]) / 2
            for p in pairs
            if counts[p.left_index] and counts[p.right_index]
        ]
    ).reshape(-1, 3)
    if len(mids) < min_pairs:
        return np.array([1.0, 0.0, 0.0]), world.mean(axis=0), len(mids)
    origin = mids.mean(axis=0)
    # Plane normal = direction of least variance of the midpoints
    normal = np.linalg.svd(mids - origin, full_matrices=False)[2][-1]
    if normal[0] < 0:
        normal = -normal
    return normal, origin, len(mids)


def _round(v: Optional[float], nd: int = 4):
    return None if v is None else round(v, nd)


def _row(fname: str, p: ROIPair, **values) -> dict:
    row = {c: None for c in QC_COLUMNS}
    row.update(
        file=fname,
        roi=p.roi,
        left_index=p.left_index,
        right_index=p.right_index,
        left_name=p.left_name,
        right_name=p.right_name,
    )
    row.update(values)
    return row


def run_qc(
    mask_paths: Iterable[str],
    out_csv: str,
    mapping_csv: Path = DEFAULT_MAPPING_CSV,
    thresholds: Optional[QCThresholds] = None,
) -> List[dict]:
    """
    Run QC on every mask and write one row per file (file|status|issues) to
    out_csv. Nothing is printed. With thresholds.detail, the per-ROI rows are
    also written to <out_csv stem>_detail.csv. Returns the per-ROI rows.
    """
    thresholds = thresholds or QCThresholds()
    pairs = load_roi_pairs(mapping_csv)
    valid_labels = load_valid_labels(mapping_csv)

    all_rows: List[dict] = []
    summary_rows: List[dict] = []
    for path in sorted(mask_paths):
        fname = os.path.basename(path)
        try:
            rows, file_issues = qc_mask(path, pairs, valid_labels, thresholds)
        except Exception as e:  # keep going; one bad file should not stop QC
            summary_rows.append(
                {"file": fname, "status": FAIL, "issues": f"could not be processed: {e}"}
            )
            continue
        all_rows.extend(rows)
        summary_rows.append(_summarize_file(fname, rows, file_issues))

    _write_csv(out_csv, QC_SUMMARY_COLUMNS, summary_rows)
    if thresholds.detail:
        detail_csv = os.path.splitext(out_csv)[0] + "_detail.csv"
        _write_csv(detail_csv, QC_COLUMNS, all_rows)
    return all_rows


def _summarize_file(fname: str, rows: List[dict], file_issues: List[str]) -> dict:
    """
    Collapse one file's results into a single file|status|issues row. The
    file-level issues come first, then every ROI issue as
    "<roi> (L:<left_index>/R:<right_index>): <issue>".
    """
    status = PASS
    for r in rows:
        status = _worse(status, r["status"])
    if "empty segmentation" in file_issues:
        return {"file": fname, "status": FAIL, "issues": "empty segmentation"}
    issues = list(file_issues) + [
        f"{r['roi']} (L:{r['left_index']}/R:{r['right_index']}): {issue}"
        for r in rows
        if r["issues"]
        for issue in r["issues"].split("; ")
    ]
    return {"file": fname, "status": status, "issues": "; ".join(issues)}


def _write_csv(path: str, columns: List[str], rows: List[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
