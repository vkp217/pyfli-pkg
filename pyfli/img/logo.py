"""
Provide logo tools for PyFLI image assets and rich-display helpers bundled with PyFLI.

This module belongs to :mod:`pyfli.img` and is part of PyFLI image assets and rich-
display helpers bundled with PyFLI. The module primarily re-exports package symbols or
constants for downstream imports.
"""

from __future__ import annotations
from typing import Any
import os
import base64

_LOGO_PATH = os.path.join(os.path.dirname(__file__), "PyFLI_logo.png")


class _LogoDisplay:
    """
    Render the packaged PyFLI logo in IPython-rich frontends. The object implements PNG
    and HTML representations while keeping the asset path private to the package.
    """

    def _repr_png_(self) -> Any:
        """
        Return the representation.

        Returns
        -------
        Any
            Return value.
        """
        with open(_LOGO_PATH, "rb") as f:
            return f.read()

    def _repr_html_(self) -> Any:
        """
        Return the representation.

        Returns
        -------
        Any
            Return value.
        """
        with open(_LOGO_PATH, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return (
            '<img src="data:image/png;base64,'
            + data
            + '" style="max-width:400px;" alt="PyFLI logo"/>'
        )

    def __repr__(self) -> str:
        """
        Return the representation.

        Returns
        -------
        str
            Return value.
        """
        return "PyFLI logo — place alone in a Jupyter cell to display"


pflogo = _LogoDisplay()
