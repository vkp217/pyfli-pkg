import h5py
import numpy as np
import pytest

from pyfli.io.detector import Detector
from pyfli.io.ss2_bin import (
    SS2_HALF_HEIGHT,
    SS2_WIDTH,
)


def _write_ss2_hdf5(
    path,
    cube: np.ndarray,
) -> None:
    with h5py.File(
        path,
        "w",
    ) as file_handle:
        group = file_handle.create_group("Gate Images")

        for gate in range(cube.shape[-1]):
            group.create_dataset(
                f"Gate {gate}",
                data=cube[..., gate],
            )


def _write_ss3_hdf5(
    path,
    cube: np.ndarray,
) -> None:
    with h5py.File(
        path,
        "w",
    ) as file_handle:
        group = file_handle.create_group("Gate Images")

        for gate in range(cube.shape[-1]):
            group.create_dataset(
                f"Bottom G2 Gate {gate}",
                data=cube[..., gate],
            )


def test_spad_detector(tmp_path):
    period_bins = 70
    phase_offset = 11

    time = np.arange(
        period_bins,
        dtype=np.float64,
    )

    period = 4.0 + 150.0 * np.exp(-time / 8.0)

    trace = np.roll(
        np.tile(
            period,
            4,
        ),
        phase_offset,
    )

    cube = np.broadcast_to(
        trace,
        (
            4,
            4,
            trace.size,
        ),
    ).astype(np.uint16)

    path = tmp_path / "spad.h5"

    with h5py.File(
        path,
        "w",
    ) as file_handle:
        dataset = file_handle.create_dataset(
            "measurement",
            data=cube,
        )

        dataset.attrs["axes"] = "YXT"

    result = Detector(data_path=str(path)).SPAD(
        config={
            "fold": True,
            "pile_up": False,
            "fold_repetitions": 4,
        }
    )

    assert result["source"] == "SPAD"

    assert result["raw_data"]["decay"].shape == (
        4,
        4,
        70,
    )

    processing = result["metadata"]["processing"]

    assert processing["sub_bg"] is False
    assert processing["pile_up"] is False
    assert processing["fold"] is True

    assert processing["spad_metadata"]["decay"]["fold"]["phase_shift"] == -11


def test_ss2_hdf5(tmp_path):
    path = tmp_path / "ss2.hdf5"

    cube = np.stack(
        [
            np.full(
                (
                    4,
                    5,
                ),
                gate,
                dtype=np.uint16,
            )
            for gate in range(3)
        ],
        axis=-1,
    )

    _write_ss2_hdf5(
        path,
        cube,
    )

    result = Detector(data_path=str(path)).SS2(
        sub_bg=False,
        pile_up=False,
        hot_pixel=False,
    )

    decay = result["raw_data"]["decay"]

    assert result["source"] == "SwissSPAD2"

    assert decay.shape == (
        4,
        5,
        3,
    )

    assert decay.dtype == np.float32

    assert result["metadata"]["processing"]["input_format"] == "hdf5"

    assert (
        result["metadata"]["processing"]["spad_metadata"]["decay"]["detector"] == "ss2"
    )

    np.testing.assert_array_equal(
        decay,
        cube,
    )


def test_ss3_hdf5(tmp_path):
    path = tmp_path / "ss3.hdf5"

    cube = np.stack(
        [
            np.full(
                (
                    4,
                    5,
                ),
                gate,
                dtype=np.uint16,
            )
            for gate in range(3)
        ],
        axis=-1,
    )

    _write_ss3_hdf5(
        path,
        cube,
    )

    result = Detector(data_path=str(path)).SS3(
        sub_bg=False,
        pile_up=False,
        hot_pixel=False,
    )

    decay = result["raw_data"]["decay"]

    assert result["source"] == "SwissSPAD3"

    assert decay.shape == (
        4,
        5,
        3,
    )

    assert decay.dtype == np.float32

    assert result["metadata"]["processing"]["input_format"] == "hdf5"

    assert (
        result["metadata"]["processing"]["spad_metadata"]["decay"]["detector"] == "ss3"
    )

    np.testing.assert_array_equal(
        decay,
        cube,
    )


def test_ss2_bin(tmp_path):
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

    result = Detector(
        data_path=str(tmp_path),
        bit_size=8,
    ).SS2(
        sub_bg=False,
        pile_up=False,
        hot_pixel=False,
        config={
            "input_format": "auto",
            "ss2_expected_gate_count": 1,
        },
    )

    decay = result["raw_data"]["decay"]

    assert result["source"] == "SwissSPAD2"

    assert result["metadata"]["processing"]["input_format"] == "ss2_bin"

    assert decay.shape == (
        512,
        512,
        1,
    )

    assert decay.dtype == np.uint16

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
        decay[
            :SS2_HALF_HEIGHT,
            :,
            0,
        ],
        expected_top,
    )

    np.testing.assert_array_equal(
        decay[
            SS2_HALF_HEIGHT:,
            :,
            0,
        ],
        expected_bottom,
    )

    np.testing.assert_array_equal(
        decay[
            SS2_HALF_HEIGHT - 1,
            :,
            0,
        ],
        decay[
            SS2_HALF_HEIGHT,
            :,
            0,
        ],
    )


def test_ss2_rejects_ss3(tmp_path):
    path = tmp_path / "ss3.hdf5"

    _write_ss3_hdf5(
        path,
        np.zeros(
            (
                4,
                5,
                3,
            ),
            dtype=np.uint16,
        ),
    )

    with pytest.raises(
        ValueError,
        match="No deterministic SPAD layout",
    ):
        Detector(data_path=str(path)).SS2(
            sub_bg=False,
            pile_up=False,
            hot_pixel=False,
        )


def test_ss3_rejects_ss2(tmp_path):
    path = tmp_path / "ss2.hdf5"

    _write_ss2_hdf5(
        path,
        np.zeros(
            (
                4,
                5,
                3,
            ),
            dtype=np.uint16,
        ),
    )

    with pytest.raises(
        ValueError,
        match="No deterministic SPAD layout",
    ):
        Detector(data_path=str(path)).SS3(
            sub_bg=False,
            pile_up=False,
            hot_pixel=False,
        )


def test_ss3_rejects_bin(tmp_path):
    frame = np.zeros(
        (
            1,
            SS2_HALF_HEIGHT,
            SS2_WIDTH,
        ),
        dtype=np.uint8,
    )

    frame.tofile(tmp_path / "top0.bin")

    frame.tofile(tmp_path / "btm0.bin")

    with pytest.raises(
        ValueError,
        match="SwissSPAD3 loading supports HDF5 input only",
    ):
        Detector(
            data_path=str(tmp_path),
            bit_size=8,
        ).SS3(
            sub_bg=False,
            pile_up=False,
            hot_pixel=False,
        )


def test_ss2_hot_pixels(tmp_path):
    data_dir = tmp_path / "data"
    background_dir = tmp_path / "background"

    data_dir.mkdir()
    background_dir.mkdir()

    data_cube = np.full(
        (
            5,
            6,
            3,
        ),
        10,
        dtype=np.uint16,
    )

    data_cube[
        2,
        3,
        :,
    ] = 200

    background_cube = np.zeros(
        (
            5,
            6,
            3,
        ),
        dtype=np.uint16,
    )

    background_cube[
        2,
        3,
        :,
    ] = 100

    _write_ss2_hdf5(
        data_dir / "frame0.hdf5",
        data_cube,
    )

    _write_ss2_hdf5(
        background_dir / "background0.hdf5",
        background_cube,
    )

    result = Detector(
        data_path=str(data_dir),
        bg_path=str(background_dir),
    ).SS2(
        sub_bg=False,
        pile_up=False,
        hot_pixel=True,
        make_hp_map=True,
        threshold_sigma=5.0,
    )

    decay = result["raw_data"]["decay"]

    assert decay.shape == (
        5,
        6,
        3,
    )

    np.testing.assert_array_equal(
        decay[
            2,
            3,
            :,
        ],
        np.array(
            [
                10,
                10,
                10,
            ]
        ),
    )

    assert result["metadata"]["processing"]["hot_pixel"] is True

    assert (
        result["metadata"]["processing"]["spad_metadata"]["decay"]["hot_pixel_scope"]
        == "per_file_before_folder_combine"
    )
