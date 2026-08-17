import h5py
import numpy as np

from pyfli.io.data_ops_static import StaticDataOps as ds
from pyfli.io.spad_io import load_spad


def test_load_spad_applies_pileup_before_circular_folding(
    tmp_path,
):
    period_bins = 70
    phase_offset = 9

    time = np.arange(
        period_bins,
        dtype=np.float64,
    )

    period = (
        5.0
        + 120.0
        * np.exp(
            -time / 10.0
        )
    )

    raw_trace = np.roll(
        np.tile(
            period,
            4,
        ),
        phase_offset,
    )

    raw_cube = np.broadcast_to(
        raw_trace,
        (
            2,
            2,
            raw_trace.size,
        ),
    ).astype(
        np.uint16
    )

    path = tmp_path / "stacked.h5"

    with h5py.File(
        path,
        "w",
    ) as file_handle:
        dataset = file_handle.create_dataset(
            "counts",
            data=raw_cube,
        )

        dataset.attrs[
            "axes"
        ] = "YXT"

    result = load_spad(
        str(path),
        {
            "bit_depth": 10,
            "pile_up": True,
            "fold": True,
            "fold_repetitions": 4,
        },
    )

    corrected = ds.pileup_correction(
        raw_cube,
        bit_size=10,
    )

    aligned = np.roll(
        corrected,
        shift=-phase_offset,
        axis=-1,
    )

    expected = aligned.reshape(
        2,
        2,
        4,
        70,
    ).sum(
        axis=-2
    )

    assert result.data.shape == (
        2,
        2,
        70,
    )

    assert result.fold_layout is not None

    assert (
        result.fold_layout.phase_shift
        == -phase_offset
    )

    np.testing.assert_allclose(
        result.data,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_load_spad_does_not_apply_pileup_when_disabled(
    tmp_path,
):
    path = tmp_path / "counts.h5"

    source = np.arange(
        4 * 4 * 8,
        dtype=np.uint16,
    ).reshape(
        4,
        4,
        8,
    )

    with h5py.File(
        path,
        "w",
    ) as file_handle:
        dataset = file_handle.create_dataset(
            "counts",
            data=source,
        )

        dataset.attrs[
            "axes"
        ] = "YXT"

    result = load_spad(
        str(path),
        {
            "pile_up": False,
            "fold": False,
        },
    )

    np.testing.assert_array_equal(
        result.data,
        source,
    )

    assert (
        result.metadata[
            "pile_up_applied"
        ]
        is False
    )

    assert (
        result.metadata[
            "fold_applied"
        ]
        is False
    )


def test_hdf5_folder_applies_pileup_before_file_sum(
    tmp_path,
):
    first = np.full(
        (
            2,
            2,
            4,
        ),
        100,
        dtype=np.uint16,
    )

    second = np.full(
        (
            2,
            2,
            4,
        ),
        200,
        dtype=np.uint16,
    )

    for index, source in enumerate(
        (
            first,
            second,
        )
    ):
        path = (
            tmp_path
            / f"frame{index}.h5"
        )

        with h5py.File(
            path,
            "w",
        ) as file_handle:
            dataset = file_handle.create_dataset(
                "counts",
                data=source,
            )

            dataset.attrs[
                "axes"
            ] = "YXT"

    result = load_spad(
        str(tmp_path),
        {
            "pile_up": True,
            "fold": False,
            "bit_depth": 10,
            "hdf5_folder_mode": "sum",
        },
    )

    expected = (
        ds.pileup_correction(
            first,
            bit_size=10,
        )
        + ds.pileup_correction(
            second,
            bit_size=10,
        )
    )

    np.testing.assert_allclose(
        result.data,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )

    assert (
        result.metadata[
            "pile_up_scope"
        ]
        == "per_file_before_folder_combine"
    )