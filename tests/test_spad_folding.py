import numpy as np
import pytest

from pyfli.io.spad_folding import (
    analyze_fold_layout,
    apply_fold_layout,
    circular_align,
)


def _periodic_decay(
    period_bins: int = 70,
    phase_offset: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(
        period_bins,
        dtype=np.float64,
    )

    period = 3.0 + 180.0 * np.exp(-time / 9.0)

    acquisition = np.roll(
        np.tile(
            period,
            4,
        ),
        phase_offset,
    )

    cube = np.empty(
        (
            2,
            3,
            acquisition.size,
        ),
        dtype=np.float64,
    )

    for row in range(cube.shape[0]):
        for col in range(cube.shape[1]):
            cube[row, col] = acquisition * (1.0 + 0.05 * row + 0.02 * col)

    return (
        period,
        cube,
    )


def test_circular_align():
    data = np.arange(6).reshape(
        1,
        1,
        6,
    )

    shifted = circular_align(
        data,
        -2,
    )

    np.testing.assert_array_equal(
        shifted.ravel(),
        np.array(
            [
                2,
                3,
                4,
                5,
                0,
                1,
            ]
        ),
    )


def test_auto_fold_phase():
    period, cube = _periodic_decay(
        period_bins=70,
        phase_offset=17,
    )

    layout = analyze_fold_layout(
        cube,
        expected_repeats=4,
    )

    folded = apply_fold_layout(
        cube,
        layout,
    )

    assert layout.period_bins == 70
    assert layout.repeat_count == 4
    assert layout.onset_index == 17
    assert layout.onset_lead_bins == 4
    assert layout.phase_origin == 13
    assert layout.phase_shift == -13
    assert layout.pulse_positions == (17, 87, 157, 227)

    assert folded.shape == (
        2,
        3,
        70,
    )

    np.testing.assert_allclose(
        folded[0, 0],
        np.roll(4.0 * period, 4),
        rtol=1e-12,
        atol=1e-12,
    )


def test_auto_fold_phase_without_lead():
    period, cube = _periodic_decay(
        period_bins=70,
        phase_offset=17,
    )

    layout = analyze_fold_layout(
        cube,
        expected_repeats=4,
        onset_lead_bins=0,
    )

    folded = apply_fold_layout(
        cube,
        layout,
    )

    assert layout.onset_index == 17
    assert layout.onset_lead_bins == 0
    assert layout.phase_origin == 17
    assert layout.phase_shift == -17

    np.testing.assert_allclose(
        folded[0, 0],
        4.0 * period,
        rtol=1e-12,
        atol=1e-12,
    )


def test_invalid_onset_lead_bins():
    _, cube = _periodic_decay(
        period_bins=70,
        phase_offset=17,
    )

    with pytest.raises(ValueError, match="onset_lead_bins"):
        analyze_fold_layout(
            cube,
            expected_repeats=4,
            onset_lead_bins=-1,
        )

    with pytest.raises(ValueError, match="onset_lead_bins"):
        analyze_fold_layout(
            cube,
            expected_repeats=4,
            period_bins=70,
            onset_lead_bins=70,
        )


def test_manual_phase_shift():
    period, cube = _periodic_decay(
        period_bins=70,
        phase_offset=2,
    )

    layout = analyze_fold_layout(
        cube,
        expected_repeats=4,
        period_bins=70,
        phase_shift=-2,
    )

    folded = apply_fold_layout(
        cube,
        layout,
    )

    assert layout.manual_period is True
    assert layout.manual_phase is True
    assert layout.onset_index == 2
    assert layout.onset_lead_bins == 0
    assert layout.phase_origin == 2
    assert layout.phase_shift == -2

    np.testing.assert_allclose(
        folded[0, 0],
        4.0 * period,
        rtol=1e-12,
        atol=1e-12,
    )


def test_invalid_manual_period():
    _, cube = _periodic_decay(
        period_bins=70,
        phase_offset=5,
    )

    with pytest.raises(
        ValueError,
        match="does not divide",
    ):
        analyze_fold_layout(
            cube,
            period_bins=71,
        )


def test_low_fold_confidence():
    rng = np.random.default_rng(42)

    cube = rng.poisson(
        3.0,
        size=(
            4,
            4,
            280,
        ),
    ).astype(np.float64)

    with pytest.raises(
        ValueError,
        match="confidence is too low",
    ):
        analyze_fold_layout(
            cube,
            expected_repeats=4,
        )
