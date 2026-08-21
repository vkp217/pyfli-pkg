"""
Discover and read SPAD data from arbitrary HDF5 structures.

This module belongs to :mod:`pyfli.io` and provides structure-aware HDF5 discovery
without requiring detector-specific group or dataset names. Explicit metadata and
user hints take priority over structural inference, while ambiguous layouts are
reported instead of selected silently.
"""

from dataclasses import asdict, dataclass
import os
import re
from typing import Any

import h5py
import numpy as np


_NUMERIC_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

_TIME_ATTRIBUTE_NAMES = (
    "gate_delay",
    "delay",
    "delay_ps",
    "delay_ns",
    "time",
    "time_ps",
    "time_ns",
    "gate_index",
    "index",
    "bin",
)

_AXIS_ATTRIBUTE_NAMES = (
    "axes",
    "axis_order",
    "dimension_order",
    "dimensions",
    "dims",
)

_TIME_AXIS_NAMES = {
    "t",
    "time",
    "gate",
    "gates",
    "bin",
    "bins",
    "delay",
}

_SPAD_KEYWORDS = (
    "gate",
    "delay",
    "time",
    "bin",
    "flim",
    "tpsf",
    "decay",
    "photon",
)


@dataclass(frozen=True)
class HDF5DatasetInfo:
    """Describe one numeric HDF5 dataset without loading its full contents."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    attributes: dict[str, Any]
    dimension_labels: tuple[str, ...]


@dataclass(frozen=True)
class HDF5Candidate:
    """Describe one candidate SPAD layout discovered inside an HDF5 file."""

    kind: str
    dataset_paths: tuple[str, ...]
    score: float
    spatial_shape: tuple[int, int] | None
    time_axis: int | None
    gate_values: tuple[float, ...] | None
    ordering_source: str | None
    original_shape: tuple[int, ...] | None

    def to_metadata(self) -> dict[str, object]:
        """Return a serializable candidate description."""
        return asdict(self)


@dataclass(frozen=True)
class SpadHDF5ReadResult:
    """Store normalized SPAD HDF5 data and discovery metadata."""

    data: np.ndarray
    candidate: HDF5Candidate
    source_path: str

    def to_metadata(self) -> dict[str, object]:
        """Return serializable metadata for the selected HDF5 layout."""
        return {
            "source_format": "hdf5",
            "source_path": self.source_path,
            "layout": self.candidate.to_metadata(),
            "output_shape": tuple(self.data.shape),
            "output_dtype": str(self.data.dtype),
        }


def _normalize_path(path: str) -> str:
    """Normalize an HDF5 object path to a leading-slash representation."""
    if not path:
        return "/"

    if path.startswith("/"):
        return path

    return f"/{path}"


def _parent_path(path: str) -> str:
    """Return the normalized HDF5 parent path."""
    normalized = _normalize_path(path)
    parent = normalized.rsplit("/", 1)[0]
    return parent or "/"


def _normalize_name(value: str) -> str:
    """Normalize an attribute or axis label for semantic comparison."""
    return re.sub(
        r"[^a-z0-9]+",
        "",
        value.lower(),
    )


def _python_scalar(value: Any) -> Any:
    """Convert HDF5 attribute values to lightweight Python representations."""
    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _python_scalar(value.item())

        return tuple(_python_scalar(item) for item in value.tolist())

    if isinstance(value, (list, tuple)):
        return tuple(_python_scalar(item) for item in value)

    return value


def _is_numeric_dataset(
    dataset: h5py.Dataset,
) -> bool:
    """Return True for non-compound numeric datasets suitable for SPAD image data."""
    if dataset.dtype.fields is not None:
        return False

    try:
        return bool(
            np.issubdtype(
                dataset.dtype,
                np.number,
            )
        )
    except TypeError:
        return False


def _collect_numeric_datasets(
    file_handle: h5py.File,
) -> list[HDF5DatasetInfo]:
    """Collect metadata for every non-compound numeric dataset in an HDF5 file."""
    datasets: list[HDF5DatasetInfo] = []

    def visitor(
        name: str,
        obj: h5py.Dataset | h5py.Group,
    ) -> None:
        if not isinstance(obj, h5py.Dataset) or not _is_numeric_dataset(obj):
            return

        attributes = {key: _python_scalar(value) for key, value in obj.attrs.items()}

        labels = tuple(str(dimension.label or "") for dimension in obj.dims)

        datasets.append(
            HDF5DatasetInfo(
                path=_normalize_path(name),
                shape=tuple(int(size) for size in obj.shape),
                dtype=str(obj.dtype),
                attributes=attributes,
                dimension_labels=labels,
            )
        )

    file_handle.visititems(visitor)
    return datasets


def _keyword_score(
    paths: tuple[str, ...],
) -> float:
    """Return a small semantic bonus without making names part of the schema."""
    text = " ".join(paths).lower()

    matches = sum(keyword in text for keyword in _SPAD_KEYWORDS)

    return float(min(matches, 5))


def _numeric_path_pattern(path: str) -> str:
    """Replace numeric tokens in a path so repeated datasets group together."""
    return _NUMERIC_PATTERN.sub(
        "{n}",
        path,
    )


def _extract_numeric_tokens(
    path: str,
) -> tuple[float, ...]:
    """Extract numeric tokens from an HDF5 path in left-to-right order."""
    return tuple(float(token) for token in _NUMERIC_PATTERN.findall(path))


def _attribute_lookup(
    attributes: dict[str, Any],
    name: str,
) -> Any | None:
    """Find an HDF5 attribute by normalized case-insensitive name."""
    target = _normalize_name(name)

    for key, value in attributes.items():
        if _normalize_name(key) == target:
            return value

    return None


def _numeric_scalar(value: Any) -> float | None:
    """Convert a scalar numeric HDF5 attribute to float, otherwise return None."""
    if isinstance(value, bool):
        return None

    if isinstance(value, (tuple, list)) and len(value) == 1:
        return _numeric_scalar(value[0])

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        numeric = float(value)

        if np.isfinite(numeric):
            return numeric

    return None


def _infer_attribute_order(
    infos: list[HDF5DatasetInfo],
    requested_attribute: str | None,
) -> (
    tuple[
        list[HDF5DatasetInfo],
        tuple[float, ...],
        str,
    ]
    | None
):
    """Infer gate ordering from a shared scalar numeric dataset attribute."""
    attribute_names = (
        (requested_attribute,)
        if requested_attribute is not None
        else _TIME_ATTRIBUTE_NAMES
    )

    for attribute_name in attribute_names:
        values = []

        for info in infos:
            value = _attribute_lookup(
                info.attributes,
                attribute_name,
            )

            numeric = _numeric_scalar(value)

            if numeric is None:
                values = []
                break

            values.append(numeric)

        if len(values) != len(infos) or len(set(values)) != len(values):
            continue

        ordered_pairs = sorted(
            zip(values, infos),
            key=lambda item: item[0],
        )

        ordered_infos = [info for _, info in ordered_pairs]

        ordered_values = tuple(float(value) for value, _ in ordered_pairs)

        return (
            ordered_infos,
            ordered_values,
            f"attribute:{attribute_name}",
        )

    return None


def _infer_path_order(
    infos: list[HDF5DatasetInfo],
) -> (
    tuple[
        list[HDF5DatasetInfo],
        tuple[float, ...],
        str,
    ]
    | None
):
    """Infer ordering from the numeric path component that varies across datasets."""
    numeric_tokens = [_extract_numeric_tokens(info.path) for info in infos]

    if not numeric_tokens or any(not tokens for tokens in numeric_tokens):
        return None

    token_count = len(numeric_tokens[0])

    if any(len(tokens) != token_count for tokens in numeric_tokens):
        return None

    varying_positions = []

    for position in range(token_count):
        values = [tokens[position] for tokens in numeric_tokens]

        if len(set(values)) == len(values):
            varying_positions.append(position)

    if not varying_positions:
        return None

    position = varying_positions[-1]

    values = [tokens[position] for tokens in numeric_tokens]

    ordered_pairs = sorted(
        zip(values, infos),
        key=lambda item: item[0],
    )

    ordered_infos = [info for _, info in ordered_pairs]

    ordered_values = tuple(float(value) for value, _ in ordered_pairs)

    return (
        ordered_infos,
        ordered_values,
        f"numeric_path_token:{position}",
    )


def _infer_split_order(
    infos: list[HDF5DatasetInfo],
    gate_order_attribute: str | None,
) -> (
    tuple[
        list[HDF5DatasetInfo],
        tuple[float, ...],
        str,
    ]
    | None
):
    """Infer deterministic gate ordering using metadata first and path numbers second."""
    attribute_order = _infer_attribute_order(
        infos,
        gate_order_attribute,
    )

    if attribute_order is not None:
        return attribute_order

    if gate_order_attribute is not None:
        return None

    return _infer_path_order(infos)


def _parse_axis_tokens(
    value: Any,
) -> list[str] | None:
    """Parse a common HDF5 axis-order attribute into normalized axis tokens."""
    value = _python_scalar(value)

    if isinstance(value, str):
        stripped = value.strip()

        if not stripped:
            return None

        if re.search(
            r"[,;/\s]",
            stripped,
        ):
            tokens = [
                token
                for token in re.split(
                    r"[,;/\s]+",
                    stripped,
                )
                if token
            ]
        elif len(stripped) <= 4:
            tokens = list(stripped)
        else:
            return None

        return [_normalize_name(token) for token in tokens]

    if isinstance(value, tuple):
        tokens = []

        for item in value:
            if not isinstance(item, str):
                return None

            tokens.append(_normalize_name(item))

        return tokens

    return None


def _infer_time_axis(
    info: HDF5DatasetInfo,
    requested_time_axis: int | None,
) -> tuple[int | None, str | None]:
    """Infer a stacked cube's temporal axis using metadata before shape clues."""
    if len(info.shape) != 3:
        return None, None

    if requested_time_axis is not None:
        axis = int(requested_time_axis)

        if axis < 0:
            axis += 3

        if axis not in (0, 1, 2):
            raise ValueError(
                f"hdf5_time_axis must resolve to 0, 1, or 2, got {requested_time_axis}."
            )

        return axis, "user_hint"

    for attribute_name in _AXIS_ATTRIBUTE_NAMES:
        value = _attribute_lookup(
            info.attributes,
            attribute_name,
        )

        if value is None:
            continue

        tokens = _parse_axis_tokens(value)

        if tokens is None or len(tokens) != 3:
            continue

        time_axes = [
            index for index, token in enumerate(tokens) if token in _TIME_AXIS_NAMES
        ]

        if len(time_axes) == 1:
            return (
                time_axes[0],
                f"attribute:{attribute_name}",
            )

    labels = [_normalize_name(label) for label in info.dimension_labels]

    time_axes = [
        index for index, label in enumerate(labels) if label in _TIME_AXIS_NAMES
    ]

    if len(time_axes) == 1:
        return (
            time_axes[0],
            "dimension_label",
        )

    first, second, third = info.shape

    if first == second and second != third:
        return (
            2,
            "shape_equal_spatial_axes",
        )

    if second == third and first != second:
        return (
            0,
            "shape_equal_spatial_axes",
        )

    if first == third and second != first:
        return (
            1,
            "shape_equal_spatial_axes",
        )

    return None, None


def _build_split_candidates(
    infos: list[HDF5DatasetInfo],
    gate_group_path: str | None,
    gate_order_attribute: str | None,
    gate_prefix: str | None,
) -> list[HDF5Candidate]:
    """Build candidate split-gate layouts from repeated 2D numeric datasets."""
    two_dimensional = [info for info in infos if len(info.shape) == 2]

    if gate_group_path is not None:
        group_path = _normalize_path(gate_group_path).rstrip("/")

        two_dimensional = [
            info
            for info in two_dimensional
            if (info.path == group_path or info.path.startswith(f"{group_path}/"))
        ]

    if gate_prefix is not None:
        two_dimensional = [
            info
            for info in two_dimensional
            if info.path.rsplit(
                "/",
                1,
            )[-1].startswith(gate_prefix)
        ]

    grouped: dict[
        tuple[object, ...],
        list[HDF5DatasetInfo],
    ] = {}

    for info in two_dimensional:
        parent_key = (
            "parent",
            _parent_path(info.path),
            info.shape,
        )

        pattern_key = (
            "pattern",
            _numeric_path_pattern(info.path),
            info.shape,
        )

        grouped.setdefault(
            parent_key,
            [],
        ).append(info)

        grouped.setdefault(
            pattern_key,
            [],
        ).append(info)

    unique_groups: dict[
        frozenset[str],
        list[HDF5DatasetInfo],
    ] = {}

    for group in grouped.values():
        if len(group) < 2:
            continue

        key = frozenset(info.path for info in group)

        unique_groups.setdefault(
            key,
            group,
        )

    candidates = []

    for group in unique_groups.values():
        ordering = _infer_split_order(
            group,
            gate_order_attribute,
        )

        if ordering is None:
            ordered_infos = sorted(
                group,
                key=lambda info: info.path,
            )
            gate_values = None
            ordering_source = None
        else:
            (
                ordered_infos,
                gate_values,
                ordering_source,
            ) = ordering

        paths = tuple(info.path for info in ordered_infos)

        score = 60.0 + min(len(paths), 20) + _keyword_score(paths)

        if ordering_source is not None:
            score += 20.0

        if gate_group_path is not None:
            score += 30.0

        if gate_prefix is not None:
            score += 30.0

        candidates.append(
            HDF5Candidate(
                kind="split",
                dataset_paths=paths,
                score=score,
                spatial_shape=tuple(int(value) for value in ordered_infos[0].shape),
                time_axis=None,
                gate_values=gate_values,
                ordering_source=ordering_source,
                original_shape=None,
            )
        )

    return candidates


def _build_stacked_candidates(
    infos: list[HDF5DatasetInfo],
    dataset_path: str | None,
    time_axis: int | None,
) -> list[HDF5Candidate]:
    """Build candidate stacked-cube layouts from 3D numeric datasets."""
    normalized_dataset_path = _normalize_path(dataset_path) if dataset_path else None

    candidates = []

    for info in infos:
        if len(info.shape) != 3:
            continue

        if normalized_dataset_path is not None and info.path != normalized_dataset_path:
            continue

        (
            detected_axis,
            axis_source,
        ) = _infer_time_axis(
            info,
            time_axis,
        )

        spatial_shape = None

        if detected_axis is not None:
            spatial_shape = tuple(
                int(size)
                for index, size in enumerate(info.shape)
                if index != detected_axis
            )

        score = 55.0 + _keyword_score((info.path,))

        if detected_axis is not None:
            score += 20.0

        if normalized_dataset_path is not None:
            score += 50.0

        candidates.append(
            HDF5Candidate(
                kind="stacked",
                dataset_paths=(info.path,),
                score=score,
                spatial_shape=spatial_shape,
                time_axis=detected_axis,
                gate_values=None,
                ordering_source=axis_source,
                original_shape=info.shape,
            )
        )

    return candidates


def inspect_spad_hdf5(
    fname: str,
    dataset_path: str | None = None,
    time_axis: int | None = None,
    gate_group_path: str | None = None,
    gate_order_attribute: str | None = None,
    gate_prefix: str | None = None,
) -> list[HDF5Candidate]:
    """
    Inspect an HDF5 file and return ranked candidate SPAD layouts.

    Parameters
    ----------
    fname : str
        HDF5 file to inspect.
    dataset_path : str | None
        Explicit stacked 3D dataset path.
    time_axis : int | None
        Explicit temporal axis for a stacked 3D dataset.
    gate_group_path : str | None
        Explicit group path containing split 2D gate datasets.
    gate_order_attribute : str | None
        Dataset attribute used to order split gates.
    gate_prefix : str | None
        Optional dataset-name prefix used as a backwards-compatible discovery hint.

    Returns
    -------
    list[HDF5Candidate]
        Candidate layouts sorted from highest to lowest discovery score.
    """
    if not fname:
        raise ValueError("HDF5 filename must be provided.")

    absolute_path = os.path.abspath(fname)

    if not os.path.isfile(absolute_path):
        raise FileNotFoundError(f"HDF5 file not found: {absolute_path}")

    with h5py.File(
        absolute_path,
        "r",
    ) as file_handle:
        infos = _collect_numeric_datasets(file_handle)

    if not infos:
        raise ValueError(f"No numeric HDF5 datasets found in: {absolute_path}")

    has_split_hint = any(
        value is not None
        for value in (
            gate_group_path,
            gate_order_attribute,
            gate_prefix,
        )
    )

    has_stacked_hint = dataset_path is not None or time_axis is not None

    split_candidates = []

    if not has_stacked_hint or has_split_hint:
        split_candidates = _build_split_candidates(
            infos,
            gate_group_path=gate_group_path,
            gate_order_attribute=gate_order_attribute,
            gate_prefix=gate_prefix,
        )

    stacked_candidates = []

    if not has_split_hint or has_stacked_hint:
        stacked_candidates = _build_stacked_candidates(
            infos,
            dataset_path=dataset_path,
            time_axis=time_axis,
        )

    return sorted(
        split_candidates + stacked_candidates,
        key=lambda candidate: candidate.score,
        reverse=True,
    )


def _candidate_is_readable(
    candidate: HDF5Candidate,
) -> bool:
    """Return whether a candidate has enough information for deterministic loading."""
    if candidate.kind == "split":
        return (
            candidate.gate_values is not None and candidate.ordering_source is not None
        )

    if candidate.kind == "stacked":
        return candidate.time_axis is not None

    return False


def _candidate_description(
    candidate: HDF5Candidate,
) -> str:
    """Return a compact human-readable HDF5 candidate description."""
    if candidate.kind == "split":
        first_path = candidate.dataset_paths[0]

        return (
            f"split[{len(candidate.dataset_paths)}] "
            f"first={first_path}, "
            f"shape={candidate.spatial_shape}, "
            f"order={candidate.ordering_source}, "
            f"score={candidate.score:.1f}"
        )

    return (
        f"stacked[{candidate.dataset_paths[0]}] "
        f"shape={candidate.original_shape}, "
        f"time_axis={candidate.time_axis}, "
        f"source={candidate.ordering_source}, "
        f"score={candidate.score:.1f}"
    )


def _select_candidate(
    candidates: list[HDF5Candidate],
) -> HDF5Candidate:
    """Select one deterministic HDF5 layout, rejecting missing or ambiguous discovery."""
    readable = [
        candidate for candidate in candidates if _candidate_is_readable(candidate)
    ]

    if not readable:
        descriptions = "\n  - ".join(
            _candidate_description(candidate) for candidate in candidates[:8]
        )

        suffix = f"\n  - {descriptions}" if descriptions else ""

        raise ValueError(
            "No deterministic SPAD layout could be inferred from the HDF5 file. "
            "Provide dataset_path/time_axis for a stacked cube, or "
            "gate_group_path/gate_order_attribute for split gate images." + suffix
        )

    readable.sort(
        key=lambda candidate: candidate.score,
        reverse=True,
    )

    if len(readable) == 1:
        return readable[0]

    best = readable[0]
    second = readable[1]

    if best.score - second.score >= 15.0:
        return best

    descriptions = "\n  - ".join(
        _candidate_description(candidate) for candidate in readable[:8]
    )

    raise ValueError(
        "Ambiguous SPAD HDF5 structure; multiple layouts are similarly plausible. "
        "Provide an explicit dataset/group hint. Candidates:\n  - " + descriptions
    )


def _read_split_candidate(
    file_handle: h5py.File,
    candidate: HDF5Candidate,
) -> np.ndarray:
    """Read an ordered set of 2D gate datasets and stack them into (H, W, T)."""
    arrays = []
    expected_shape = candidate.spatial_shape

    for path in candidate.dataset_paths:
        array = np.asarray(file_handle[path][...])

        if array.ndim != 2:
            raise ValueError(f"Split SPAD dataset '{path}' is not 2D: {array.shape}.")

        if expected_shape is not None and tuple(array.shape) != expected_shape:
            raise ValueError(
                f"Split SPAD dataset '{path}' has shape "
                f"{array.shape}, expected {expected_shape}."
            )

        arrays.append(array)

    return np.stack(
        arrays,
        axis=-1,
    )


def _read_stacked_candidate(
    file_handle: h5py.File,
    candidate: HDF5Candidate,
) -> np.ndarray:
    """Read one stacked 3D dataset and move its temporal axis to the last dimension."""
    path = candidate.dataset_paths[0]

    array = np.asarray(file_handle[path][...])

    if array.ndim != 3:
        raise ValueError(f"Stacked SPAD dataset '{path}' is not 3D: {array.shape}.")

    if candidate.time_axis is None:
        raise ValueError(
            f"Temporal axis for stacked SPAD dataset '{path}' is unresolved."
        )

    return np.moveaxis(
        array,
        candidate.time_axis,
        -1,
    )


def read_spad_hdf5(
    fname: str,
    dataset_path: str | None = None,
    time_axis: int | None = None,
    gate_group_path: str | None = None,
    gate_order_attribute: str | None = None,
    gate_prefix: str | None = None,
) -> SpadHDF5ReadResult:
    """
    Discover and normalize SPAD image data from an HDF5 file into (H, W, T).

    Parameters
    ----------
    fname : str
        HDF5 file to read.
    dataset_path : str | None
        Explicit stacked 3D dataset path when automatic discovery is ambiguous.
    time_axis : int | None
        Explicit temporal axis for a stacked 3D dataset.
    gate_group_path : str | None
        Explicit group path containing split 2D gate datasets.
    gate_order_attribute : str | None
        Dataset attribute used to order split gate datasets.
    gate_prefix : str | None
        Optional dataset-name prefix used as a backwards-compatible discovery hint.

    Returns
    -------
    SpadHDF5ReadResult
        Normalized HDF5 data cube and discovery metadata.
    """
    absolute_path = os.path.abspath(fname)

    candidates = inspect_spad_hdf5(
        absolute_path,
        dataset_path=dataset_path,
        time_axis=time_axis,
        gate_group_path=gate_group_path,
        gate_order_attribute=gate_order_attribute,
        gate_prefix=gate_prefix,
    )

    candidate = _select_candidate(candidates)

    with h5py.File(
        absolute_path,
        "r",
    ) as file_handle:
        if candidate.kind == "split":
            data = _read_split_candidate(
                file_handle,
                candidate,
            )

        elif candidate.kind == "stacked":
            data = _read_stacked_candidate(
                file_handle,
                candidate,
            )

        else:
            raise ValueError(f"Unsupported HDF5 SPAD candidate kind: {candidate.kind}")

    if data.ndim != 3:
        raise ValueError(f"SPAD HDF5 output must be 3D (H, W, T), got {data.shape}.")

    return SpadHDF5ReadResult(
        data=data,
        candidate=candidate,
        source_path=absolute_path,
    )
