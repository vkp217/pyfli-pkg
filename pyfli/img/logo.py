"""
Provide logo tools for PyFLI image assets and rich-display helpers bundled with PyFLI.

This module belongs to :mod:`pyfli.img` and is part of PyFLI image assets and rich-
display helpers bundled with PyFLI. The module primarily re-exports package symbols or
constants for downstream imports.
"""

import base64
import os
from typing import Any

_LOGO_PATH = os.path.join(os.path.dirname(__file__), "PyFLI_logo.png")


class _LogoDisplay:
    """
    Render the packaged PyFLI logo in IPython-rich frontends. The object implements PNG
    and HTML representations while keeping the asset path private to the package.
    """

    def _repr_png_(self) -> Any:
        """
        Run the repr png routine.

        Returns
        -------
        Any
            Object produced by repr png.
        """
        with open(_LOGO_PATH, "rb") as f:
            return f.read()

    def _repr_html_(self) -> Any:
        """
        Run the repr html routine.

        Returns
        -------
        Any
            Object produced by repr html.
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
        Run the repr routine.

        Returns
        -------
        str
            String path, label, or message produced by repr.
        """
        return "PyFLI logo — place alone in a Jupyter cell to display"


pflogo = _LogoDisplay()
