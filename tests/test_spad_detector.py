import h5py
import numpy as np

from pyfli.io.detector import Detector


def test_detector_spad_packages_generic_hdf5_and_fold_metadata(
    tmp_path,
):
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


def test_existing_ss2_detector_method_still_reads_gate_images(
    tmp_path,
):
    path = tmp_path / "ss2.hdf5"

    with h5py.File(
        path,
        "w",
    ) as file_handle:
        group = file_handle.create_group("Gate Images")

        for gate in range(3):
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

    result = Detector(data_path=str(path)).SS2(
        sub_bg=False,
        pile_up=False,
        hot_pixel=False,
    )

    assert result["raw_data"]["decay"].shape == (
        4,
        5,
        3,
    )


def test_existing_ss3_detector_method_still_reads_gate_images(
    tmp_path,
):
    path = tmp_path / "ss3.hdf5"

    with h5py.File(
        path,
        "w",
    ) as file_handle:
        group = file_handle.create_group("Gate Images")

        for gate in range(3):
            group.create_dataset(
                f"Bottom G2 Gate {gate}",
                data=np.full(
                    (
                        4,
                        5,
                    ),
                    gate,
                    dtype=np.uint16,
                ),
            )

    result = Detector(data_path=str(path)).SS3(
        sub_bg=False,
        pile_up=False,
        hot_pixel=False,
    )

    assert result["raw_data"]["decay"].shape == (
        4,
        5,
        3,
    )
