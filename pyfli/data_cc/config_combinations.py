"""
Cartesian-product config generator with optional weighted sampling.

"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


class ConfigCombinationGenerator:
    """
    Parameters
    ----------
    base_config : dict
        Full config dict. Keys not listed in `sweep` are held fixed
        across every generated combination.
    sweep : dict[str, list]
        Maps a config key to the list of values it should take on, e.g.
            {"jitter": [True, False], "n_cycles": [1e5, 5e5, 1e6, 2e6]}
        Total combinations = product of len(v) over all sweep keys.
    weights : dict[str, list[float]], optional
        Per-key sampling weights, same order/length as the matching
        `sweep` list. Keys omitted here default to uniform weighting.
        A combination's weight is the product of its per-key weights
        (i.e. keys are treated as independent), then renormalized to
        sum to 1. Only affects `.sample()` — `.all_combinations()`
        is always exhaustive and order-preserving regardless of weights.
    combo_overrides : dict[tuple, float], optional
        Escape hatch for when independence isn't good enough: maps a
        specific tuple of values (in `sweep` key order, e.g.
        (True, 1e6)) to a weight multiplier applied on top of the
        per-key product weight for that exact combination.
    """

    def __init__(
        self,
        base_config: dict[str, Any],
        sweep: dict[str, Sequence[Any]],
        weights: dict[str, Sequence[float]] | None = None,
        combo_overrides: dict[tuple[Any, ...], float] | None = None,
    ):
        if not sweep:
            raise ValueError("sweep must contain at least one key")

        self.base_config = dict(base_config)
        self.sweep_keys: list[str] = list(sweep.keys())
        self.sweep_values: list[list[Any]] = [list(v) for v in sweep.values()]
        self.weights = weights or {}
        self.combo_overrides = combo_overrides or {}

        for k, vals in zip(self.sweep_keys, self.sweep_values):
            w = self.weights.get(k)
            if w is not None and len(w) != len(vals):
                raise ValueError(
                    f"weights['{k}'] has length {len(w)}, expected {len(vals)} "
                    f"to match sweep['{k}']"
                )

        # every combination as a row of per-key indices, shape (n_combos, n_keys)
        self._combo_indices = np.array(
            list(itertools.product(*[range(len(v)) for v in self.sweep_values])),
            dtype=int,
        )

        # per-key weight vectors (uniform default), combined multiplicatively
        per_key_w = [
            np.ones(len(vals))
            if self.weights.get(k) is None
            else np.asarray(self.weights[k], dtype=float)
            for k, vals in zip(self.sweep_keys, self.sweep_values)
        ]
        combo_w = np.ones(len(self._combo_indices))
        for j, w_arr in enumerate(per_key_w):
            combo_w *= w_arr[self._combo_indices[:, j]]

        # apply explicit overrides, if any
        if self.combo_overrides:
            for row_i, idx_row in enumerate(self._combo_indices):
                key = tuple(
                    self.sweep_values[j][idx_row[j]] for j in range(len(idx_row))
                )
                if key in self.combo_overrides:
                    combo_w[row_i] *= self.combo_overrides[key]

        if combo_w.sum() <= 0:
            raise ValueError(
                "combination weights sum to zero — check `weights`/`combo_overrides`"
            )
        self._combo_probs = combo_w / combo_w.sum()

    @property
    def n_combinations(self) -> int:
        return len(self._combo_indices)

    def _build_config(self, idx_row: np.ndarray) -> dict[str, Any]:
        cfg = dict(self.base_config)
        for k, vlist, i in zip(self.sweep_keys, self.sweep_values, idx_row):
            cfg[k] = vlist[i]
        return cfg

    def all_combinations(self) -> Iterable[dict[str, Any]]:
        """Yield every combination exactly once (deterministic, exhaustive)."""
        for idx_row in self._combo_indices:
            yield self._build_config(idx_row)

    def combination_table(self) -> tuple[list[dict[str, Any]], np.ndarray]:
        """Return (all configs, their sampling probabilities) — for inspection."""
        return [self._build_config(r) for r in self._combo_indices], self._combo_probs

    def sample(
        self, n: int = 1, seed: int | None = None, replace: bool = True
    ) -> list[dict[str, Any]]:
        """Draw n configs according to the weighted distribution."""
        rng = np.random.default_rng(seed)
        chosen = rng.choice(
            self.n_combinations, size=n, replace=replace, p=self._combo_probs
        )
        return [self._build_config(self._combo_indices[i]) for i in chosen]

    def __len__(self) -> int:
        return self.n_combinations

    def __repr__(self) -> str:
        return (
            f"ConfigCombinationGenerator(sweep_keys={self.sweep_keys}, "
            f"n_combinations={self.n_combinations})"
        )
