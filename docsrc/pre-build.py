import logging
import os
import shutil
import sys
from pathlib import Path

try:
    from sphinx_polyversion.git import Git

    USE_POLYVERSION = True
except ImportError:
    USE_POLYVERSION = False

logging.basicConfig()
logger = logging.getLogger("pre-build.py")
logger.setLevel(logging.DEBUG)


def patch_conf(sourcedir):
    root = (
        Git.root(Path(__file__).parent)
        if USE_POLYVERSION
        else str(Path(os.path.abspath(sourcedir)).parent)
    )
    sourcedir = Path(sourcedir)
    cursrc = Path(root) / "docsrc"
    if os.path.abspath(cursrc) == os.path.abspath(sourcedir):
        return
    conf_src = cursrc / "conf.py"
    conf_dst = sourcedir / "conf.py"
    if conf_src.exists():
        logger.info("Overwriting old conf.py with current conf.py")
        shutil.copy2(conf_src, conf_dst)
    # If future conf.py changes reference NEW static assets that older
    # revisions don't have on disk, list them here so they get copied
    # forward too, e.g.:
    # for rel in ["../pyfli/img/PyFLI_logo_light.png"]:
    #     ...


if __name__ == "__main__":
    logger.info("Running pre-build script")
    patch_conf(sys.argv[1])
