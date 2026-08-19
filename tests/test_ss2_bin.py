import os

import numpy as np
import pytest

from pyfli.io.ss2_bin import (
    SS2_HALF_HEIGHT,
    SS2_WIDTH,
    combine_ss2_10bit_subframes,
    deinterleave_ss2_columns,
    discover_ss2_bin_files,
    read_ss2_bin_acquisition,
    read_ss2_bin_file,
)


def test_column_deinterleave():
    raw = np.zeros(
        (
            1,
            SS2_HALF_HEIGHT,
            SS2_WIDTH,
        ),
        dtype=np.uint8,
    )

    raw[0, 0, 0] = 11
    raw[0, 0, 128] = 22
    raw[0, 0, 256] = 33
    raw[0, 0, 384] = 44

    physical = deinterleave_ss2_columns(raw)

    np.testing.assert_array_equal(
        physical[
            0,
            0,
            :4,
        ],
        np.array(
            [
                11,
                22,
                33,
                44,
            ]
        ),
    )


def test_10bit_subframes():
    frames = np.empty(
        (
            4,
            SS2_HALF_HEIGHT,
            SS2_WIDTH,
        ),
        dtype=np.uint8,
    )

    frames[0].fill(1)
    frames[1].fill(2)
    frames[2].fill(4)
    frames[3].fill(8)

    decoded = combine_ss2_10bit_subframes(frames)

    assert decoded.dtype == np.uint16

    assert decoded.shape == (
        1,
        SS2_HALF_HEIGHT,
        SS2_WIDTH,
    )

    assert np.all(decoded == 15)


def test_bin_10bit_decode(tmp_path):
    frames = np.empty(
        (
            4,
            SS2_HALF_HEIGHT,
            SS2_WIDTH,
        ),
        dtype=np.uint8,
    )

    frames[0].fill(1)
    frames[1].fill(2)
    frames[2].fill(3)
    frames[3].fill(4)

    path = tmp_path / "top0.bin"

    frames.tofile(path)

    decoded = read_ss2_bin_file(
        str(path),
        bit_depth=10,
    )

    assert decoded.shape == (
        SS2_HALF_HEIGHT,
        SS2_WIDTH,
        1,
    )

    assert decoded.dtype == np.uint16

    assert np.all(decoded == 10)


def test_bin_stitch(tmp_path):
    row_values = np.arange(
        SS2_HALF_HEIGHT,
        dtype=np.uint8,
    ).reshape(
        1,
        SS2_HALF_HEIGHT,
        1,
    )

    top = np.broadcast_to(
        row_values,
        (
            1,
            SS2_HALF_HEIGHT,
            SS2_WIDTH,
        ),
    ).copy()

    bottom = np.broadcast_to(
        row_values,
        (
            1,
            SS2_HALF_HEIGHT,
            SS2_WIDTH,
        ),
    ).copy()

    top.tofile(tmp_path / "top0.bin")

    bottom.tofile(tmp_path / "btm0.bin")

    result = read_ss2_bin_acquisition(
        str(tmp_path),
        bit_depth=8,
        expected_gate_count=1,
    )

    assert result.data.shape == (
        512,
        512,
        1,
    )

    assert result.data.dtype == np.uint16

    expected_top = np.broadcast_to(
        np.arange(
            SS2_HALF_HEIGHT,
            dtype=np.uint16,
        ).reshape(
            SS2_HALF_HEIGHT,
            1,
        ),
        (
            SS2_HALF_HEIGHT,
            SS2_WIDTH,
        ),
    )

    expected_bottom = np.flip(
        expected_top,
        axis=0,
    )

    np.testing.assert_array_equal(
        result.data[
            :SS2_HALF_HEIGHT,
            :,
            0,
        ],
        expected_top,
    )

    np.testing.assert_array_equal(
        result.data[
            SS2_HALF_HEIGHT:,
            :,
            0,
        ],
        expected_bottom,
    )

    np.testing.assert_array_equal(
        result.data[
            SS2_HALF_HEIGHT - 1,
            :,
            0,
        ],
        result.data[
            SS2_HALF_HEIGHT,
            :,
            0,
        ],
    )


def test_bin_chunk_order(tmp_path):
    frame = np.zeros(
        (
            1,
            SS2_HALF_HEIGHT,
            SS2_WIDTH,
        ),
        dtype=np.uint8,
    )

    for index in range(11):
        frame.tofile(tmp_path / f"top{index}.bin")

        frame.tofile(tmp_path / f"btm{index}.bin")

    (
        top_files,
        bottom_files,
        indices,
    ) = discover_ss2_bin_files(str(tmp_path))

    assert indices == tuple(range(11))

    assert os.path.basename(top_files[2]) == "top2.bin"

    assert os.path.basename(top_files[10]) == "top10.bin"

    assert os.path.basename(bottom_files[2]) == "btm2.bin"

    assert os.path.basename(bottom_files[10]) == "btm10.bin"


def test_bin_pair_mismatch(tmp_path):
    frame = np.zeros(
        (
            1,
            SS2_HALF_HEIGHT,
            SS2_WIDTH,
        ),
        dtype=np.uint8,
    )

    frame.tofile(tmp_path / "top0.bin")

    frame.tofile(tmp_path / "top1.bin")

    frame.tofile(tmp_path / "btm0.bin")

    with pytest.raises(
        ValueError,
        match="chunk sets do not match",
    ):
        read_ss2_bin_acquisition(
            str(tmp_path),
            bit_depth=8,
        )


def test_bin_missing_chunk(tmp_path):
    frame = np.zeros(
        (
            1,
            SS2_HALF_HEIGHT,
            SS2_WIDTH,
        ),
        dtype=np.uint8,
    )

    for index in (
        0,
        2,
    ):
        frame.tofile(tmp_path / f"top{index}.bin")

        frame.tofile(tmp_path / f"btm{index}.bin")

    with pytest.raises(
        ValueError,
        match="not contiguous",
    ):
        read_ss2_bin_acquisition(
            str(tmp_path),
            bit_depth=8,
        )
