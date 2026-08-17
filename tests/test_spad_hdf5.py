import h5py
import numpy as np
import pytest

from pyfli.io.spad_hdf5 import read_spad_hdf5


def test_reads_existing_ss2_gate_structure_in_numeric_order(
    tmp_path,
):
    path = tmp_path / "ss2.hdf5"

    with h5py.File(
        path,
        "w",
    ) as file_handle:
        group = file_handle.create_group("Gate Images")

        for gate in (
            1,
            10,
            2,
        ):
            group.create_dataset(
                f"Gate {gate}",
                data=np.full(
                    (
                        4,
                        5,
                    ),
                    gate,
                    dtype=np.uint16,
                ),
            )

    result = read_spad_hdf5(
        str(path),
        gate_prefix="Gate ",
    )

    assert result.data.shape == (
        4,
        5,
        3,
    )

    np.testing.assert_array_equal(
        result.data[
            0,
            0,
        ],
        np.array(
            [
                1,
                2,
                10,
            ]
        ),
    )


def test_reads_existing_ss3_gate_structure_in_numeric_order(
    tmp_path,
):
    path = tmp_path / "ss3.hdf5"

    with h5py.File(
        path,
        "w",
    ) as file_handle:
        group = file_handle.create_group("Gate Images")

        for gate in (
            2,
            0,
            1,
        ):
            group.create_dataset(
                f"Bottom G2 Gate {gate}",
                data=np.full(
                    (
                        3,
                        4,
                    ),
                    gate,
                    dtype=np.uint16,
                ),
            )

    result = read_spad_hdf5(
        str(path),
        gate_prefix="Bottom G2 Gate",
    )

    assert result.data.shape == (
        3,
        4,
        3,
    )

    np.testing.assert_array_equal(
        result.data[
            0,
            0,
        ],
        np.array(
            [
                0,
                1,
                2,
            ]
        ),
    )


def test_reads_arbitrary_nested_gate_names_from_attributes(
    tmp_path,
):
    path = tmp_path / "custom.h5"

    with h5py.File(
        path,
        "w",
    ) as file_handle:
        for delay in (
            100,
            0,
            50,
        ):
            group = file_handle.create_group(f"acquisition/position_{delay}")

            dataset = group.create_dataset(
                "image",
                data=np.full(
                    (
                        3,
                        5,
                    ),
                    delay,
                    dtype=np.uint16,
                ),
            )

            dataset.attrs["delay_ps"] = delay

    result = read_spad_hdf5(str(path))

    assert result.data.shape == (
        3,
        5,
        3,
    )

    np.testing.assert_array_equal(
        result.data[
            0,
            0,
        ],
        np.array(
            [
                0,
                50,
                100,
            ]
        ),
    )

    assert result.candidate.ordering_source == "attribute:delay_ps"


def test_reads_stacked_cube_using_axes_metadata(
    tmp_path,
):
    path = tmp_path / "stacked.h5"

    source = np.arange(
        5 * 3 * 4,
        dtype=np.uint16,
    ).reshape(
        5,
        3,
        4,
    )

    with h5py.File(
        path,
        "w",
    ) as file_handle:
        dataset = file_handle.create_dataset(
            "measurement",
            data=source,
        )

        dataset.attrs["axes"] = "TYX"

    result = read_spad_hdf5(str(path))

    assert result.data.shape == (
        3,
        4,
        5,
    )

    np.testing.assert_array_equal(
        result.data,
        np.moveaxis(
            source,
            0,
            -1,
        ),
    )


def test_rejects_ambiguous_multiple_gate_groups(
    tmp_path,
):
    path = tmp_path / "ambiguous.h5"

    with h5py.File(
        path,
        "w",
    ) as file_handle:
        for group_name in (
            "raw",
            "background",
        ):
            group = file_handle.create_group(group_name)

            for gate in range(3):
                group.create_dataset(
                    f"gate_{gate}",
                    data=np.full(
                        (
                            4,
                            4,
                        ),
                        gate,
                        dtype=np.uint16,
                    ),
                )

    with pytest.raises(
        ValueError,
        match="Ambiguous SPAD HDF5 structure",
    ):
        read_spad_hdf5(str(path))


def test_rejects_stacked_cube_when_time_axis_is_ambiguous(
    tmp_path,
):
    path = tmp_path / "ambiguous_axes.h5"

    with h5py.File(
        path,
        "w",
    ) as file_handle:
        file_handle.create_dataset(
            "measurement",
            data=np.zeros(
                (
                    8,
                    8,
                    8,
                ),
                dtype=np.uint16,
            ),
        )

    with pytest.raises(
        ValueError,
        match="No deterministic SPAD layout",
    ):
        read_spad_hdf5(str(path))
