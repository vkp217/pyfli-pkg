# IRF Deconvolution

Low-level gate-matrix / cyclic-convolution based deconvolution solver, with detector-specific parameter presets (TCSPC, SPAD, ICCD).

!!! info "Import path"
    These symbols are not re-exported at the top-level `pyfli` package — import them from `pyfli.scripts`:

    ```python
    from pyfli.scripts import solve_flim, SolverConfig
    ```

## Detector parameter presets

::: pyfli.scripts.TCSPCParams

::: pyfli.scripts.SPADParams

::: pyfli.scripts.ICCDParams

## Solver

::: pyfli.scripts.SolverConfig

::: pyfli.scripts.solve_flim

## Observation model

::: pyfli.scripts.make_observation

::: pyfli.scripts.generalized_anscombe

## Gate matrix & convolution helpers

::: pyfli.scripts.build_gate_matrix

::: pyfli.scripts.decay_basis

::: pyfli.scripts.cyclic_conv
