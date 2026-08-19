"""
Decode native SwissSPAD2 binary acquisitions into PyFLI image cubes.

This module belongs to :mod:`pyfli.io` and implements the SwissSPAD2 binary layout
used by topN.bin / btmN.bin acquisitions. Raw 256 x 512 detector halves are decoded,
column banks are deinterleaved, 10-bit subframes are accumulated, and top/bottom
halves are stitched into a 512 x 512 x T data cube.
"""

from dataclasses import dataclass
import os
import re

import numpy as np


SS2_HALF_HEIGHT = 256
SS2_WIDTH = 512
SS2_COLUMN_BANKS = 4
SS2_BANK_WIDTH = SS2_WIDTH // SS2_COLUMN_BANKS
SS2_10BIT_SUBFRAMES = 4
SS2_RAW_DTYPE = np.uint8
SS2_DECODE_BLOCK_GATES = 64


@dataclass(frozen=True)
class SS2BinReadResult:
    """
    Store a decoded SwissSPAD2 binary acquisition and its source metadata.

    Parameters
    ----------
    data : np.ndarray
        Stitched SwissSPAD2 data cube with shape (512, 512, T).
    bit_depth : int
        Acquisition bit depth used to decode raw binary frames.
    chunk_indices : tuple[int, ...]
        Numeric chunk indices discovered for the top and bottom binary files.
    top_files : tuple[str, ...]
        Ordered top-detector source files.
    bottom_files : tuple[str, ...]
        Ordered bottom-detector source files.
    raw_frame_count : int
        Total number of raw binary frames per detector half.
    gate_count : int
        Number of decoded acquisition gates before any temporal folding.
    """

    data: np.ndarray
    bit_depth: int
    chunk_indices: tuple[int, ...]
    top_files: tuple[str, ...]
    bottom_files: tuple[str, ...]
    raw_frame_count: int
    gate_count: int

    def to_metadata(self) -> dict[str, object]:
        """
        Convert the binary-read result to serializable metadata.

        Returns
        -------
        dict[str, object]
            Dictionary describing the decoded SwissSPAD2 binary acquisition.
        """
        return {
            "source_format": "ss2_bin",
            "bit_depth": self.bit_depth,
            "chunk_indices": self.chunk_indices,
            "top_files": self.top_files,
            "bottom_files": self.bottom_files,
            "raw_frame_count": self.raw_frame_count,
            "gate_count": self.gate_count,
            "half_shape": (SS2_HALF_HEIGHT, SS2_WIDTH),
            "stitched_shape": tuple(self.data.shape),
            "column_banks": SS2_COLUMN_BANKS,
            "bank_width": SS2_BANK_WIDTH,
            "raw_dtype": np.dtype(SS2_RAW_DTYPE).name,
            "output_dtype": str(self.data.dtype),
            "subframes_per_gate": (SS2_10BIT_SUBFRAMES if self.bit_depth == 10 else 1),
        }


def _validate_bit_depth(bit_depth: int) -> None:
    """Validate SwissSPAD2 binary acquisition bit depth."""
    if bit_depth not in (8, 10):
        raise ValueError(
            f"SwissSPAD2 binary decoding supports bit_depth 8 or 10, got {bit_depth}."
        )


def _compile_chunk_pattern(prefix: str) -> re.Pattern[str]:
    """Compile a case-insensitive top/bottom chunk filename pattern."""
    if not prefix:
        raise ValueError("SwissSPAD2 binary filename prefix cannot be empty.")

    return re.compile(
        rf"^{re.escape(prefix)}(?P<index>\d+)\.bin$",
        re.IGNORECASE,
    )


def discover_ss2_bin_files(
    path: str,
    top_prefix: str = "top",
    bottom_prefix: str = "btm",
) -> tuple[list[str], list[str], tuple[int, ...]]:
    """
    Discover and numerically order matching SwissSPAD2 top/bottom binary chunks.

    Parameters
    ----------
    path : str
        Directory containing the acquisition or one top/bottom .bin file inside it.
    top_prefix : str
        Filename prefix used for the top detector half.
    bottom_prefix : str
        Filename prefix used for the bottom detector half.

    Returns
    -------
    tuple[list[str], list[str], tuple[int, ...]]
        Ordered top files, ordered bottom files, and their common numeric chunk indices.
    """
    if not path:
        raise ValueError("SwissSPAD2 binary path must be provided.")

    absolute_path = os.path.abspath(path)

    if not os.path.exists(absolute_path):
        raise FileNotFoundError(f"SwissSPAD2 binary path not found: {absolute_path}")

    if os.path.isdir(absolute_path):
        folder_path = absolute_path
    else:
        if not absolute_path.lower().endswith(".bin"):
            raise ValueError(
                f"SwissSPAD2 binary input must be a .bin file or directory, got: {path}"
            )

        folder_path = os.path.dirname(absolute_path)

    top_pattern = _compile_chunk_pattern(top_prefix)
    bottom_pattern = _compile_chunk_pattern(bottom_prefix)

    top_by_index: dict[int, str] = {}
    bottom_by_index: dict[int, str] = {}

    for filename in os.listdir(folder_path):
        top_match = top_pattern.match(filename)
        bottom_match = bottom_pattern.match(filename)

        if top_match:
            index = int(top_match.group("index"))

            if index in top_by_index:
                raise ValueError(
                    f"Duplicate top chunk index {index} in SwissSPAD2 acquisition."
                )

            top_by_index[index] = os.path.join(
                folder_path,
                filename,
            )

        elif bottom_match:
            index = int(bottom_match.group("index"))

            if index in bottom_by_index:
                raise ValueError(
                    f"Duplicate bottom chunk index {index} in SwissSPAD2 acquisition."
                )

            bottom_by_index[index] = os.path.join(
                folder_path,
                filename,
            )

    if not top_by_index:
        raise FileNotFoundError(
            f"No '{top_prefix}N.bin' SwissSPAD2 files found in: {folder_path}"
        )

    if not bottom_by_index:
        raise FileNotFoundError(
            f"No '{bottom_prefix}N.bin' SwissSPAD2 files found in: {folder_path}"
        )

    top_indices = set(top_by_index)
    bottom_indices = set(bottom_by_index)

    if top_indices != bottom_indices:
        missing_top = sorted(bottom_indices - top_indices)
        missing_bottom = sorted(top_indices - bottom_indices)

        details = []

        if missing_top:
            details.append(f"missing top chunks {missing_top}")

        if missing_bottom:
            details.append(f"missing bottom chunks {missing_bottom}")

        raise ValueError(
            "SwissSPAD2 top/bottom chunk sets do not match: " + ", ".join(details)
        )

    chunk_indices = tuple(sorted(top_indices))

    expected_indices = set(
        range(
            chunk_indices[0],
            chunk_indices[-1] + 1,
        )
    )

    if set(chunk_indices) != expected_indices:
        missing_indices = sorted(expected_indices - set(chunk_indices))

        raise ValueError(
            "SwissSPAD2 binary chunk indices are not contiguous; "
            f"missing chunks {missing_indices}."
        )

    top_files = [top_by_index[index] for index in chunk_indices]

    bottom_files = [bottom_by_index[index] for index in chunk_indices]

    if os.path.isfile(absolute_path):
        selected_name = os.path.basename(absolute_path)

        selected_match = top_pattern.match(selected_name) or bottom_pattern.match(
            selected_name
        )

        if selected_match is None:
            raise ValueError(
                f"Binary file '{selected_name}' does not match "
                f"'{top_prefix}N.bin' or '{bottom_prefix}N.bin'; "
                "the acquisition pairing cannot be inferred."
            )

    return (
        top_files,
        bottom_files,
        chunk_indices,
    )


def _raw_frame_count(file_path: str) -> int:
    """Return the number of complete 256 x 512 uint8 frames in one BIN chunk."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"SwissSPAD2 binary file not found: {file_path}")

    byte_count = os.path.getsize(file_path)

    frame_bytes = SS2_HALF_HEIGHT * SS2_WIDTH * np.dtype(SS2_RAW_DTYPE).itemsize

    if byte_count == 0:
        raise ValueError(f"SwissSPAD2 binary file is empty: {file_path}")

    if byte_count % frame_bytes != 0:
        raise ValueError(
            f"SwissSPAD2 binary file '{os.path.basename(file_path)}' "
            f"contains {byte_count} bytes, which is not divisible by "
            f"one {SS2_HALF_HEIGHT} x {SS2_WIDTH} raw frame "
            f"({frame_bytes} bytes)."
        )

    return byte_count // frame_bytes


def _decoded_gate_count(
    file_path: str,
    bit_depth: int,
) -> tuple[int, int]:
    """Return raw-frame and decoded-gate counts for one SwissSPAD2 BIN chunk."""
    _validate_bit_depth(bit_depth)

    raw_frames = _raw_frame_count(file_path)

    subframes_per_gate = SS2_10BIT_SUBFRAMES if bit_depth == 10 else 1

    if raw_frames % subframes_per_gate != 0:
        raise ValueError(
            f"SwissSPAD2 file '{os.path.basename(file_path)}' "
            f"contains {raw_frames} raw frames, which is not divisible "
            f"by {subframes_per_gate} for bit_depth={bit_depth}."
        )

    return (
        raw_frames,
        raw_frames // subframes_per_gate,
    )


def deinterleave_ss2_columns(
    raw_frames: np.ndarray,
) -> np.ndarray:
    """
    Convert SwissSPAD2 four-bank raw column order into physical detector column order.

    Parameters
    ----------
    raw_frames : np.ndarray
        Array whose last two dimensions are (256, 512), with raw columns stored as four
        consecutive 128-column banks.

    Returns
    -------
    np.ndarray
        Array with the same shape and dtype, reordered into physical detector columns.
    """
    frames = np.asarray(raw_frames)

    if frames.ndim < 2 or frames.shape[-2:] != (SS2_HALF_HEIGHT, SS2_WIDTH):
        raise ValueError(
            "SwissSPAD2 column deinterleaving requires trailing dimensions "
            f"({SS2_HALF_HEIGHT}, {SS2_WIDTH}), got {frames.shape}."
        )

    leading_shape = frames.shape[:-2]

    banked = frames.reshape(
        *leading_shape,
        SS2_HALF_HEIGHT,
        SS2_COLUMN_BANKS,
        SS2_BANK_WIDTH,
    )

    deinterleaved = np.swapaxes(
        banked,
        -2,
        -1,
    )

    return deinterleaved.reshape(
        *leading_shape,
        SS2_HALF_HEIGHT,
        SS2_WIDTH,
    )


def combine_ss2_10bit_subframes(
    frames: np.ndarray,
) -> np.ndarray:
    """
    Sum each group of four SwissSPAD2 raw subframes into one 10-bit acquisition gate.

    Parameters
    ----------
    frames : np.ndarray
        Deinterleaved raw frames with shape (N, 256, 512).

    Returns
    -------
    np.ndarray
        Decoded acquisition gates with shape (N / 4, 256, 512) and uint16 dtype.
    """
    array = np.asarray(frames)

    if array.ndim != 3 or array.shape[1:] != (SS2_HALF_HEIGHT, SS2_WIDTH):
        raise ValueError(
            "SwissSPAD2 10-bit accumulation requires shape "
            f"(N, {SS2_HALF_HEIGHT}, {SS2_WIDTH}), "
            f"got {array.shape}."
        )

    if array.shape[0] % SS2_10BIT_SUBFRAMES != 0:
        raise ValueError(
            f"SwissSPAD2 10-bit data contains {array.shape[0]} raw frames; "
            f"the count must be divisible by {SS2_10BIT_SUBFRAMES}."
        )

    gate_count = array.shape[0] // SS2_10BIT_SUBFRAMES

    grouped = array.reshape(
        gate_count,
        SS2_10BIT_SUBFRAMES,
        SS2_HALF_HEIGHT,
        SS2_WIDTH,
    )

    return np.sum(
        grouped,
        axis=1,
        dtype=np.uint16,
    )


def _decode_ss2_bin_into(
    file_path: str,
    target: np.ndarray,
    bit_depth: int,
) -> tuple[int, int]:
    """Decode one BIN chunk blockwise into a preallocated (256, 512, T) target view."""
    raw_frame_count, gate_count = _decoded_gate_count(
        file_path,
        bit_depth,
    )

    expected_shape = (
        SS2_HALF_HEIGHT,
        SS2_WIDTH,
        gate_count,
    )

    if target.shape != expected_shape:
        raise ValueError(
            f"SwissSPAD2 decode target has shape {target.shape}, "
            f"expected {expected_shape}."
        )

    if target.dtype != np.uint16:
        raise ValueError(
            f"SwissSPAD2 decode target must have uint16 dtype, got {target.dtype}."
        )

    subframes_per_gate = SS2_10BIT_SUBFRAMES if bit_depth == 10 else 1

    raw_map = np.memmap(
        file_path,
        dtype=SS2_RAW_DTYPE,
        mode="r",
    )

    raw_frames = raw_map.reshape(
        raw_frame_count,
        SS2_HALF_HEIGHT,
        SS2_WIDTH,
    )

    for gate_start in range(
        0,
        gate_count,
        SS2_DECODE_BLOCK_GATES,
    ):
        gate_stop = min(
            gate_start + SS2_DECODE_BLOCK_GATES,
            gate_count,
        )

        raw_start = gate_start * subframes_per_gate
        raw_stop = gate_stop * subframes_per_gate

        raw_block = raw_frames[raw_start:raw_stop]

        physical_block = deinterleave_ss2_columns(raw_block)

        if bit_depth == 10:
            decoded_block = combine_ss2_10bit_subframes(physical_block)
        else:
            decoded_block = physical_block.astype(
                np.uint16,
                copy=False,
            )

        target[
            ...,
            gate_start:gate_stop,
        ] = np.moveaxis(
            decoded_block,
            0,
            -1,
        )

    del raw_frames
    del raw_map

    return (
        raw_frame_count,
        gate_count,
    )


def read_ss2_bin_file(
    file_path: str,
    bit_depth: int = 10,
) -> np.ndarray:
    """
    Decode one SwissSPAD2 top or bottom binary chunk into (256, 512, T).

    Parameters
    ----------
    file_path : str
        Path to one SwissSPAD2 binary chunk.
    bit_depth : int
        Acquisition bit depth. Supported values are 8 and 10.

    Returns
    -------
    np.ndarray
        Decoded detector-half cube with shape (256, 512, T) and uint16 dtype.
    """
    _, gate_count = _decoded_gate_count(
        file_path,
        bit_depth,
    )

    decoded = np.empty(
        (
            SS2_HALF_HEIGHT,
            SS2_WIDTH,
            gate_count,
        ),
        dtype=np.uint16,
    )

    _decode_ss2_bin_into(
        file_path,
        decoded,
        bit_depth,
    )

    return decoded


def read_ss2_bin_acquisition(
    path: str,
    bit_depth: int = 10,
    expected_gate_count: int | None = None,
    top_prefix: str = "top",
    bottom_prefix: str = "btm",
) -> SS2BinReadResult:
    """
    Decode, orient, concatenate, and stitch a complete SwissSPAD2 binary acquisition.

    The native SwissSPAD2 top and bottom detector halves use opposite row
    orientations. The top half is retained in decoded row order, while the bottom
    half is vertically flipped before it is placed below the top half. The resulting
    cube is returned in physical detector orientation with shape (512, 512, T).

    Parameters
    ----------
    path : str
        Directory containing topN.bin / btmN.bin files or one file within that set.
    bit_depth : int
        Acquisition bit depth. Supported values are 8 and 10.
    expected_gate_count : int | None
        Optional expected number of decoded gates before temporal folding.
    top_prefix : str
        Filename prefix used for the top detector half.
    bottom_prefix : str
        Filename prefix used for the bottom detector half.

    Returns
    -------
    SS2BinReadResult
        Stitched 512 x 512 x T cube and exact source/decode metadata.
    """
    _validate_bit_depth(bit_depth)

    if expected_gate_count is not None and expected_gate_count < 1:
        raise ValueError(
            "expected_gate_count must be >= 1 when provided, "
            f"got {expected_gate_count}."
        )

    (
        top_files,
        bottom_files,
        chunk_indices,
    ) = discover_ss2_bin_files(
        path,
        top_prefix=top_prefix,
        bottom_prefix=bottom_prefix,
    )

    chunk_gate_counts = []
    total_raw_frames = 0
    total_gate_count = 0

    for (
        chunk_index,
        top_file,
        bottom_file,
    ) in zip(
        chunk_indices,
        top_files,
        bottom_files,
    ):
        (
            top_raw_frames,
            top_gates,
        ) = _decoded_gate_count(
            top_file,
            bit_depth,
        )

        (
            bottom_raw_frames,
            bottom_gates,
        ) = _decoded_gate_count(
            bottom_file,
            bit_depth,
        )

        if top_raw_frames != bottom_raw_frames:
            raise ValueError(
                f"SwissSPAD2 chunk {chunk_index} has "
                f"{top_raw_frames} top raw frames but "
                f"{bottom_raw_frames} bottom raw frames."
            )

        if top_gates != bottom_gates:
            raise ValueError(
                f"SwissSPAD2 chunk {chunk_index} decodes to "
                f"{top_gates} top gates but "
                f"{bottom_gates} bottom gates."
            )

        chunk_gate_counts.append(top_gates)

        total_raw_frames += top_raw_frames
        total_gate_count += top_gates

    if expected_gate_count is not None and total_gate_count != expected_gate_count:
        raise ValueError(
            f"SwissSPAD2 decoded {total_gate_count} gates, "
            f"expected {expected_gate_count}."
        )

    stitched = np.empty(
        (
            SS2_HALF_HEIGHT * 2,
            SS2_WIDTH,
            total_gate_count,
        ),
        dtype=np.uint16,
    )

    gate_offset = 0

    for (
        top_file,
        bottom_file,
        chunk_gate_count,
    ) in zip(
        top_files,
        bottom_files,
        chunk_gate_counts,
    ):
        gate_slice = slice(
            gate_offset,
            gate_offset + chunk_gate_count,
        )

        top_target = stitched[
            :SS2_HALF_HEIGHT,
            :,
            gate_slice,
        ]

        _decode_ss2_bin_into(
            top_file,
            top_target,
            bit_depth,
        )

        bottom_target = stitched[
            SS2_HALF_HEIGHT:,
            :,
            gate_slice,
        ]

        _decode_ss2_bin_into(
            bottom_file,
            np.flip(
                bottom_target,
                axis=0,
            ),
            bit_depth,
        )

        gate_offset += chunk_gate_count

    return SS2BinReadResult(
        data=stitched,
        bit_depth=bit_depth,
        chunk_indices=chunk_indices,
        top_files=tuple(top_files),
        bottom_files=tuple(bottom_files),
        raw_frame_count=total_raw_frames,
        gate_count=total_gate_count,
    )
