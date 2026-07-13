# Supported Data Formats

`pyfli` provides native import support for several fluorescence lifetime imaging acquisition systems, unified behind a common [`DataOperations`](../api/data_io.md) interface.

| System | Acquisition type | Import helper |
|---|---|---|
| ICCD (Intensified CCD) | Fast-gated, wide-field imaging | [`AlliGprocessedImport`](../api/data_io.md) |
| SwissSPAD2 / SwissSPAD3 | Single-Photon Avalanche Diode photon counting | [`Detector`](../api/data_io.md), [`spAnalysis`](../api/sp_analysis.md) |
| Becker & Hickl (SPCImage/TCSPC) | Time-Correlated Single Photon Counting microscopy | [`BHprocessedImport`](../api/data_io.md) |
| pyfli-processed sessions | Previously processed/saved `pyfli` results | [`PyFliprocessedImport`](../api/data_io.md) |

Each importer normalizes its source format into the same decay-cube representation used throughout the rest of the package (fitting, phasor analysis, simulation, and visualization), so downstream code does not need to know which hardware produced the data.

See the [Data I/O API reference](../api/data_io.md) for the full parameter list of each importer.
