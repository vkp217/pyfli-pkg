import logging
import os
import shutil
import sys
from pathlib import Path

try:
    from sphinx_polyversion import load
    from sphinx_polyversion.git import Git, GitRefType

    USE_POLYVERSION = True
except ImportError:
    USE_POLYVERSION = False

VERSIONED_ENTRIES = {"api"}
SKIPPED_ENTRIES = {
    "_build",
    "_build_polyversion",
    ".docs_venvs",
    "jupyter_execute",
    "__pycache__",
    "conf.py",
}

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
    if _building_tag():
        _use_latest_pages(cursrc, sourcedir)


def _building_tag():
    """True when polyversion is building a release tag (not a branch)."""
    if not USE_POLYVERSION:
        return False
    try:
        return load()["current"].type_ == GitRefType.TAG
    except Exception:
        return False


def _remove(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _use_latest_pages(cursrc, sourcedir):
    """
    Replace every non-API entry of a tag's `sourcedir` with the one from the
    current `cursrc`, so a release version shows the latest guide, example and
    other pages while its API reference (``api/``) is still generated from
    that release's own code. Symlinks (e.g. ``examples -> ../examples``) are
    copied as the content they point to, so the latest notebooks are used.
    """
    latest = {
        p.name
        for p in cursrc.iterdir()
        if p.name not in VERSIONED_ENTRIES | SKIPPED_ENTRIES
    }
    for old in sourcedir.iterdir():
        if (
            old.name not in VERSIONED_ENTRIES | SKIPPED_ENTRIES
            and old.name not in latest
        ):
            logger.info(f"Removing {old.name} (not in the latest docs)")
            _remove(old)
    for name in sorted(latest):
        src, dst = cursrc / name, sourcedir / name
        _remove(dst)
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=False)
        else:
            shutil.copy2(src, dst)
        logger.info(f"Using latest {name}")


if __name__ == "__main__":
    logger.info("Running pre-build script")
    patch_conf(sys.argv[1])
