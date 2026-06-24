import os
import base64

_LOGO_PATH = os.path.join(os.path.dirname(__file__), "PyFLI_logo.png")


class _LogoDisplay:
    """Singleton that renders the PyFLI logo via IPython's rich-display protocol.

    In a Jupyter notebook, place ``pflogo`` alone on the last line of a cell
    (no parentheses needed) and the logo appears as an inline image.
    """

    def _repr_png_(self):
        with open(_LOGO_PATH, "rb") as f:
            return f.read()

    def _repr_html_(self):
        with open(_LOGO_PATH, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return (
            '<img src="data:image/png;base64,'
            + data
            + '" style="max-width:400px;" alt="PyFLI logo"/>'
        )

    def __repr__(self):
        return "PyFLI logo — place alone in a Jupyter cell to display"


pflogo = _LogoDisplay()
