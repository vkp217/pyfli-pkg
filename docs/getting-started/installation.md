# Installation

Install the stable version directly from PyPI:

```bash
pip install pyfli-lib
```

!!! note "Import name vs. package name"
    Even though the package is installed as `pyfli-lib`, you import it as `pyfli` in your scripts:

    ```python
    import pyfli
    ```

## GPU support

For users requiring GPU-based processing, install the optional tensor/AI dependencies:

```bash
pip install "pyfli-lib[gpu]"
```

## TensorFlow support

Some deep-learning-based inference paths use TensorFlow. Install it as an optional extra:

```bash
pip install "pyfli-lib[tf]"
```

## Development install

To work on `pyfli` itself (running the test suite, contributing changes):

```bash
git clone https://github.com/vkp217/pyfli-pkg.git
cd pyfli-pkg
pip install -e ".[dev]"
```

This pulls in `pytest` and `black` alongside the core dependencies. Run the test suite with:

```bash
python -m pytest
```

## Requirements

- Python 3.11 or newer
- Core dependencies (installed automatically): NumPy, SciPy, pandas, scikit-learn, scikit-image, OpenCV, matplotlib, seaborn, PyTorch, h5py, tifffile, sdtfile, and PySide6 for GUI tooling.
