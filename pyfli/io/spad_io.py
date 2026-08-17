"""
Coordinate generic SPAD file loading, optional pile-up correction, and temporal folding.

This module belongs to :mod:`pyfli.io` and provides one normalized (H, W, T) import
path for generic SPAD HDF5 files and native SwissSPAD2 binary acquisitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import re
from typing import Any

import numpy as np

from .data_ops_static import StaticDataOps as ds
from .spad_folding import (
    SpadFoldLayout,
    analyze_fold_layout,
    apply_fold_layout,
)
from .spad_hdf5 import read_spad_hdf5
from .ss2_bin import read_ss2_bin_acquisition


@dataclass
class SpadConfig:
    """
    Validate and store options for generic SPAD data import.

    Parameters
    ----------
    input_format : str
        Input format selector: 'auto', 'hdf5', or 'ss2_bin'.
    bit_depth : int
        Detector digitization bit depth used for optional pile-up correction.
    pile_up : bool
        Whether to apply PyFLI photon-counter pile-up correction before folding.
    fold : bool
        Whether repeated excitation periods should be circularly aligned and summed.
    detector_frequency_mhz : float | None
        Detector acquisition/synchronization frequency used to constrain repeat count.
    laser_frequency_mhz : float | None
        Laser repetition frequency used to constrain repeat count.
    fold_repetitions : int | None
        Explicit expected number of excitation periods in the acquisition.
    period_bins : int | None
        Explicit number of gates in one excitation period.
    phase_shift : int | None
        Explicit temporal circular shift. None enables automatic phase detection.
    min_fold_confidence : float
        Minimum confidence accepted for automatic fold detection.
    fold_validate : bool
        Whether low-confidence automatic folding should raise an error.
    period_search_radius : float
        Fractional search radius around an expected period.
    fold_smoothing_sigma : float
        Circular Gaussian smoothing sigma used only for timing detection.
    onset_threshold_fraction : float
        Peak-to-baseline fraction used by the onset detector.
    hdf5_dataset_path : str | None
        Explicit stacked HDF5 dataset path.
    hdf5_time_axis : int | None
        Explicit temporal axis for a stacked HDF5 dataset.
    hdf5_gate_group_path : str | None
        Explicit HDF5 group containing split 2D gate datasets.
    hdf5_gate_order_attribute : str | None
        HDF5 dataset attribute used to order split gate datasets.
    hdf5_gate_prefix : str | None
        Optional split-gate dataset prefix used as a discovery hint.
    hdf5_folder_mode : str
        Combination mode for directories containing multiple HDF5 cubes.
    ss2_expected_gate_count : int | None
        Optional expected SwissSPAD2 gate count before folding.
    ss2_top_prefix : str
        Filename prefix for SwissSPAD2 top-detector chunks.
    ss2_bottom_prefix : str
        Filename prefix for SwissSPAD2 bottom-detector chunks.
    """

    input_format: str = "auto"
    bit_depth: int = 10
    pile_up: bool = False
    fold: bool = False

    detector_frequency_mhz: float | None = None
    laser_frequency_mhz: float | None = None
    fold_repetitions: int | None = None

    period_bins: int | None = None
    phase_shift: int | None = None

    min_fold_confidence: float = 0.60
    fold_validate: bool = True
    period_search_radius: float = 0.15
    fold_smoothing_sigma: float = 1.0
    onset_threshold_fraction: float = 0.10

    hdf5_dataset_path: str | None = None
    hdf5_time_axis: int | None = None
    hdf5_gate_group_path: str | None = None
    hdf5_gate_order_attribute: str | None = None
    hdf5_gate_prefix: str | None = None
    hdf5_folder_mode: str = "sum"

    ss2_expected_gate_count: int | None = None
    ss2_top_prefix: str = "top"
    ss2_bottom_prefix: str = "btm"

    def __post_init__(self) -> None:
        """Validate SPAD import configuration values."""
        self.input_format = (
            self.input_format
            .lower()
            .replace("-", "_")
        )

        aliases = {
            "auto": "auto",
            "h5": "hdf5",
            "hdf5": "hdf5",
            "bin": "ss2_bin",
            "ss2": "ss2_bin",
            "ss2_bin": "ss2_bin",
        }

        if self.input_format not in aliases:
            raise ValueError(
                "input_format must be one of "
                "'auto', 'hdf5', or 'ss2_bin', "
                f"got '{self.input_format}'."
            )

        self.input_format = aliases[
            self.input_format
        ]

        if self.bit_depth < 1:
            raise ValueError(
                f"bit_depth must be >= 1, "
                f"got {self.bit_depth}."
            )

        if (
            self.detector_frequency_mhz is not None
            and self.detector_frequency_mhz <= 0
        ):
            raise ValueError(
                "detector_frequency_mhz must be positive when provided."
            )

        if (
            self.laser_frequency_mhz is not None
            and self.laser_frequency_mhz <= 0
        ):
            raise ValueError(
                "laser_frequency_mhz must be positive when provided."
            )

        if (
            self.fold_repetitions is not None
            and self.fold_repetitions < 2
        ):
            raise ValueError(
                "fold_repetitions must be >= 2 when provided."
            )

        if (
            self.period_bins is not None
            and self.period_bins < 2
        ):
            raise ValueError(
                "period_bins must be >= 2 when provided."
            )

        if not (
            0
            <= self.min_fold_confidence
            <= 1
        ):
            raise ValueError(
                "min_fold_confidence must be in [0, 1]."
            )

        if not (
            0
            < self.period_search_radius
            <= 0.5
        ):
            raise ValueError(
                "period_search_radius must be in (0, 0.5]."
            )

        if self.fold_smoothing_sigma < 0:
            raise ValueError(
                "fold_smoothing_sigma must be >= 0."
            )

        if not (
            0
            < self.onset_threshold_fraction
            < 1
        ):
            raise ValueError(
                "onset_threshold_fraction must be in (0, 1)."
            )

        if self.hdf5_folder_mode not in (
            "sum",
            "mean",
        ):
            raise ValueError(
                "hdf5_folder_mode must be 'sum' or 'mean'."
            )

        if (
            self.ss2_expected_gate_count is not None
            and self.ss2_expected_gate_count < 1
        ):
            raise ValueError(
                "ss2_expected_gate_count must be >= 1 when provided."
            )

        if (
            not self.ss2_top_prefix
            or not self.ss2_bottom_prefix
        ):
            raise ValueError(
                "SwissSPAD2 top/bottom filename prefixes cannot be empty."
            )

    @classmethod
    def from_value(
        cls,
        value: SpadConfig | dict[str, Any] | None,
        default_bit_depth: int = 10,
    ) -> SpadConfig:
        """
        Build a validated configuration from an existing object, mapping, or defaults.

        Parameters
        ----------
        value : SpadConfig | dict[str, Any] | None
            Existing configuration or user-provided configuration dictionary.
        default_bit_depth : int
            Bit depth inherited from Detector when the mapping does not specify one.

        Returns
        -------
        SpadConfig
            Validated SPAD import configuration.
        """
        if isinstance(value, cls):
            return value

        if value is None:
            return cls(
                bit_depth=default_bit_depth
            )

        if not isinstance(value, dict):
            raise TypeError(
                "SPAD config must be SpadConfig, "
                "dict[str, Any], or None, "
                f"got {type(value).__name__}."
            )

        valid_fields = set(
            cls.__dataclass_fields__
        )

        unknown = sorted(
            set(value)
            - valid_fields
        )

        if unknown:
            raise ValueError(
                f"Unknown SPAD config keys: {unknown}"
            )

        config_values = dict(value)

        config_values.setdefault(
            "bit_depth",
            default_bit_depth,
        )

        return cls(
            **config_values
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return the validated configuration as serializable metadata."""
        return asdict(self)


@dataclass(frozen=True)
class SpadReadResult:
    """Store normalized SPAD data, processing metadata, and optional fold layout."""

    data: np.ndarray
    metadata: dict[str, Any]
    fold_layout: SpadFoldLayout | None = None


def _natural_sort_key(
    value: str,
) -> list[int | str]:
    """Return a natural-sort key so numeric file chunks are ordered numerically."""
    parts = re.split(
        r"(\d+)",
        value.lower(),
    )

    return [
        int(part)
        if part.isdigit()
        else part
        for part in parts
    ]


def _resolve_expected_repeats(
    config: SpadConfig,
) -> int | None:
    """Resolve repeated-period count from an explicit value or physical frequencies."""
    if config.fold_repetitions is not None:
        return config.fold_repetitions

    if (
        config.detector_frequency_mhz is None
        and config.laser_frequency_mhz is None
    ):
        return None

    if (
        config.detector_frequency_mhz is None
        or config.laser_frequency_mhz is None
    ):
        raise ValueError(
            "Both detector_frequency_mhz and laser_frequency_mhz "
            "are required when frequency-based folding is requested."
        )

    ratio = (
        config.laser_frequency_mhz
        / config.detector_frequency_mhz
    )

    repetitions = int(
        round(ratio)
    )

    if (
        repetitions < 2
        or not np.isclose(
            ratio,
            repetitions,
            rtol=0.02,
            atol=0.02,
        )
    ):
        raise ValueError(
            "laser_frequency_mhz / detector_frequency_mhz must be "
            "an integer repeat ratio >= 2 for folding, "
            f"got {ratio:.6f}."
        )

    return repetitions


def _detect_input_format(
    path: str,
    configured_format: str,
) -> str:
    """Determine the SPAD input format unless explicitly configured."""
    if configured_format != "auto":
        return configured_format

    absolute_path = os.path.abspath(
        path
    )

    if os.path.isfile(absolute_path):
        extension = os.path.splitext(
            absolute_path
        )[1].lower()

        if extension in (
            ".h5",
            ".hdf5",
        ):
            return "hdf5"

        if extension == ".bin":
            return "ss2_bin"

        raise ValueError(
            f"Unsupported SPAD file extension: "
            f"'{extension}'."
        )

    if not os.path.isdir(absolute_path):
        raise FileNotFoundError(
            f"SPAD input path not found: {absolute_path}"
        )

    filenames = os.listdir(
        absolute_path
    )

    has_bin = any(
        filename.lower().endswith(".bin")
        for filename in filenames
    )

    has_hdf5 = any(
        filename.lower().endswith(
            (
                ".h5",
                ".hdf5",
            )
        )
        for filename in filenames
    )

    if has_bin and has_hdf5:
        raise ValueError(
            "SPAD input directory contains both BIN and HDF5 files. "
            "Set input_format explicitly to 'ss2_bin' or 'hdf5'."
        )

    if has_bin:
        return "ss2_bin"

    if has_hdf5:
        return "hdf5"

    raise FileNotFoundError(
        f"No supported SPAD BIN or HDF5 files found in: "
        f"{absolute_path}"
    )


def _hdf5_reader_kwargs(
    config: SpadConfig,
) -> dict[str, Any]:
    """Return HDF5 discovery hints from a SPAD configuration."""
    return {
        "dataset_path": config.hdf5_dataset_path,
        "time_axis": config.hdf5_time_axis,
        "gate_group_path": config.hdf5_gate_group_path,
        "gate_order_attribute": config.hdf5_gate_order_attribute,
        "gate_prefix": config.hdf5_gate_prefix,
    }


def _load_hdf5_path(
    path: str,
    config: SpadConfig,
) -> tuple[
    np.ndarray,
    dict[str, Any],
]:
    """Load one HDF5 file or combine a directory of normalized HDF5 SPAD cubes."""
    absolute_path = os.path.abspath(
        path
    )

    reader_kwargs = _hdf5_reader_kwargs(
        config
    )

    if os.path.isfile(absolute_path):
        result = read_spad_hdf5(
            absolute_path,
            **reader_kwargs,
        )

        return (
            result.data,
            result.to_metadata(),
        )

    if not os.path.isdir(absolute_path):
        raise FileNotFoundError(
            f"SPAD HDF5 path not found: {absolute_path}"
        )

    filenames = sorted(
        (
            filename
            for filename in os.listdir(
                absolute_path
            )
            if filename.lower().endswith(
                (
                    ".h5",
                    ".hdf5",
                )
            )
        ),
        key=_natural_sort_key,
    )

    if not filenames:
        raise FileNotFoundError(
            f"No HDF5 files found in: {absolute_path}"
        )

    first_result = read_spad_hdf5(
        os.path.join(
            absolute_path,
            filenames[0],
        ),
        **reader_kwargs,
    )

    first_data = first_result.data

    accumulator_dtype = (
        np.uint64
        if np.issubdtype(
            first_data.dtype,
            np.integer,
        )
        else np.float64
    )

    accumulator = first_data.astype(
        accumulator_dtype,
        copy=True,
    )

    file_metadata = [
        first_result.to_metadata()
    ]

    for filename in filenames[1:]:
        result = read_spad_hdf5(
            os.path.join(
                absolute_path,
                filename,
            ),
            **reader_kwargs,
        )

        if result.data.shape != first_data.shape:
            raise ValueError(
                "HDF5 folder contains mismatched SPAD shapes: "
                f"{first_data.shape} and {result.data.shape} "
                f"in '{filename}'."
            )

        if (
            np.issubdtype(
                accumulator.dtype,
                np.integer,
            )
            and not np.issubdtype(
                result.data.dtype,
                np.integer,
            )
        ):
            accumulator = accumulator.astype(
                np.float64
            )
            accumulator_dtype = np.float64

        accumulator += result.data.astype(
            accumulator_dtype,
            copy=False,
        )

        file_metadata.append(
            result.to_metadata()
        )

    if config.hdf5_folder_mode == "mean":
        data = (
            accumulator.astype(
                np.float64,
                copy=False,
            )
            / len(filenames)
        )
    else:
        data = accumulator

    metadata = {
        "source_format": "hdf5_folder",
        "source_path": absolute_path,
        "folder_mode": config.hdf5_folder_mode,
        "file_count": len(filenames),
        "files": file_metadata,
        "output_shape": tuple(data.shape),
        "output_dtype": str(data.dtype),
    }

    return (
        data,
        metadata,
    )


def _load_hdf5_folder_with_pileup(
    path: str,
    config: SpadConfig,
) -> np.ndarray:
    """Load an HDF5 folder with pile-up correction before file combination."""
    absolute_path = os.path.abspath(
        path
    )

    if not os.path.isdir(absolute_path):
        raise ValueError(
            "Per-file HDF5 pile-up processing requires a directory input, "
            f"got: {absolute_path}"
        )

    filenames = sorted(
        (
            filename
            for filename in os.listdir(
                absolute_path
            )
            if filename.lower().endswith(
                (
                    ".h5",
                    ".hdf5",
                )
            )
        ),
        key=_natural_sort_key,
    )

    if not filenames:
        raise FileNotFoundError(
            f"No HDF5 files found in: {absolute_path}"
        )

    reader_kwargs = _hdf5_reader_kwargs(
        config
    )

    accumulator = None
    reference_shape = None

    for filename in filenames:
        result = read_spad_hdf5(
            os.path.join(
                absolute_path,
                filename,
            ),
            **reader_kwargs,
        )

        if reference_shape is None:
            reference_shape = result.data.shape

        elif result.data.shape != reference_shape:
            raise ValueError(
                "HDF5 folder contains mismatched SPAD shapes: "
                f"{reference_shape} and {result.data.shape} "
                f"in '{filename}'."
            )

        corrected = ds.pileup_correction(
            result.data,
            bit_size=config.bit_depth,
        )

        if accumulator is None:
            accumulator = corrected.astype(
                np.float64,
                copy=True,
            )
        else:
            accumulator += corrected.astype(
                np.float64,
                copy=False,
            )

    if accumulator is None:
        raise RuntimeError(
            "HDF5 folder pile-up accumulation did not produce data."
        )

    if config.hdf5_folder_mode == "mean":
        accumulator /= len(filenames)

    return accumulator.astype(
        np.float32
    )


def _load_raw_spad(
    path: str,
    config: SpadConfig,
) -> tuple[
    np.ndarray,
    dict[str, Any],
    str,
]:
    """Load SPAD input without pile-up correction or temporal folding."""
    if not path:
        raise ValueError(
            "SPAD input path must be provided."
        )

    absolute_path = os.path.abspath(
        path
    )

    if not os.path.exists(absolute_path):
        raise FileNotFoundError(
            f"SPAD input path not found: {absolute_path}"
        )

    input_format = _detect_input_format(
        absolute_path,
        config.input_format,
    )

    if input_format == "hdf5":
        data, metadata = _load_hdf5_path(
            absolute_path,
            config,
        )

    elif input_format == "ss2_bin":
        result = read_ss2_bin_acquisition(
            absolute_path,
            bit_depth=config.bit_depth,
            expected_gate_count=config.ss2_expected_gate_count,
            top_prefix=config.ss2_top_prefix,
            bottom_prefix=config.ss2_bottom_prefix,
        )

        data = result.data
        metadata = result.to_metadata()

    else:
        raise ValueError(
            f"Unsupported SPAD input format: "
            f"{input_format}"
        )

    if data.ndim != 3:
        raise ValueError(
            f"SPAD reader must return (H, W, T), "
            f"got shape {data.shape}."
        )

    return (
        data,
        metadata,
        input_format,
    )


def load_spad(
    path: str,
    config: SpadConfig | dict[str, Any] | None = None,
    default_bit_depth: int = 10,
    fold_layout: SpadFoldLayout | None = None,
) -> SpadReadResult:
    """
    Load SPAD data, optionally apply pile-up correction, then align and fold periods.

    Parameters
    ----------
    path : str
        SPAD HDF5 path, SwissSPAD2 BIN path, or supported acquisition directory.
    config : SpadConfig | dict[str, Any] | None
        Validated configuration or configuration dictionary.
    default_bit_depth : int
        Detector bit depth used when config does not provide one.
    fold_layout : SpadFoldLayout | None
        Existing fold layout to reuse for a related IRF/background acquisition.

    Returns
    -------
    SpadReadResult
        Normalized SPAD cube and complete import/processing metadata.
    """
    resolved_config = SpadConfig.from_value(
        config,
        default_bit_depth=default_bit_depth,
    )

    (
        raw_data,
        source_metadata,
        input_format,
    ) = _load_raw_spad(
        path,
        resolved_config,
    )

    raw_shape = tuple(
        raw_data.shape
    )

    detected_layout = (
        fold_layout
        if resolved_config.fold
        else None
    )

    if resolved_config.fold:
        if detected_layout is None:
            expected_repeats = _resolve_expected_repeats(
                resolved_config
            )

            detected_layout = analyze_fold_layout(
                raw_data,
                expected_repeats=expected_repeats,
                period_bins=resolved_config.period_bins,
                phase_shift=resolved_config.phase_shift,
                min_confidence=resolved_config.min_fold_confidence,
                validate=resolved_config.fold_validate,
                search_radius=resolved_config.period_search_radius,
                smoothing_sigma=resolved_config.fold_smoothing_sigma,
                threshold_fraction=resolved_config.onset_threshold_fraction,
            )

        elif (
            raw_data.shape[-1]
            != detected_layout.original_bins
        ):
            raise ValueError(
                f"Reused fold layout expects "
                f"{detected_layout.original_bins} temporal gates, "
                f"but '{path}' has {raw_data.shape[-1]}."
            )

    processed = raw_data
    pile_up_scope = None

    if resolved_config.pile_up:
        if (
            input_format == "hdf5"
            and source_metadata.get("source_format")
            == "hdf5_folder"
        ):
            processed = _load_hdf5_folder_with_pileup(
                path,
                resolved_config,
            )

            pile_up_scope = (
                "per_file_before_folder_combine"
            )

        else:
            processed = ds.pileup_correction(
                processed,
                bit_size=resolved_config.bit_depth,
            )

            pile_up_scope = "loaded_cube"

    if resolved_config.fold:
        if detected_layout is None:
            raise RuntimeError(
                "SPAD fold layout was not resolved."
            )

        processed = apply_fold_layout(
            processed,
            detected_layout,
        )

    metadata = {
        "source": source_metadata,
        "input_format": input_format,
        "raw_shape": raw_shape,
        "raw_dtype": str(raw_data.dtype),
        "output_shape": tuple(processed.shape),
        "output_dtype": str(processed.dtype),
        "bit_depth": resolved_config.bit_depth,
        "pile_up_applied": resolved_config.pile_up,
        "pile_up_scope": pile_up_scope,
        "fold_applied": resolved_config.fold,
        "fold": (
            detected_layout.to_metadata()
            if detected_layout is not None
            else None
        ),
        "config": resolved_config.to_metadata(),
    }

    return SpadReadResult(
        data=processed,
        metadata=metadata,
        fold_layout=detected_layout,
    )


class SpadIO:
    """Provide a class-style entry point for generic SPAD import operations."""

    @staticmethod
    def load(
        path: str,
        config: SpadConfig | dict[str, Any] | None = None,
        default_bit_depth: int = 10,
        fold_layout: SpadFoldLayout | None = None,
    ) -> SpadReadResult:
        """Load SPAD data through the shared generic import pipeline."""
        return load_spad(
            path,
            config=config,
            default_bit_depth=default_bit_depth,
            fold_layout=fold_layout,
        )