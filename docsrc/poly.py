from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent))
from polyversion_patches import (
    DynamicPip,
    CustomDriver,
    PyDataVersionEncoder,
    version_key,
    visible_versions,
)

from sphinx_polyversion.api import apply_overrides
from sphinx_polyversion.git import Git, GitRef, GitRefType, file_predicate
from sphinx_polyversion.pyvenv import Environment, VenvWrapper
from sphinx_polyversion.sphinx import SphinxBuilder, Placeholder
from datetime import datetime

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

root = Git.root(Path(__file__).parent)

#: Branches to build docs for
BRANCH_REGEX = r"^(main|dev)$"

#: Tags to build docs for (starts matching once you tag releases, e.g. v0.2.0)
TAG_REGEX = r"^v[0-9]+\.[0-9]+\.[0-9]+$"

OUTPUT_DIR = "_build_polyversion"
SOURCE_DIR = "docsrc"

MOCK_DATA = {
    "revisions": [],
    "current": GitRef("local", "", "", GitRefType.BRANCH, datetime.now()),
}
MOCK = False
SEQUENTIAL = False
SPHINX_ARGS = "-a -v"

#: Doc build deps (kept in sync with pyproject.toml's `docs` extra)
SPHINX_DEPS = [
    "sphinx>=7.0",
    "pydata-sphinx-theme",
    "myst-nb",
    "sphinx-copybutton",
    "sphinx-design",
    "sphinx-polyversion==1.1.0",
]

VENV_DIR_NAME = ".docs_venvs"


def data(driver, rev, env):
    revisions = driver.targets
    return {
        "current": rev,
        "revisions": revisions,
        "latest": max(revisions, key=version_key),
    }


def root_data(driver):
    all_revisions = driver.builds
    latest = max(all_revisions, key=version_key)
    # Root page only lists the same capped set as the version switcher;
    # `latest` stays uncapped so the redirect always targets the true latest.
    return {"revisions": visible_versions(all_revisions), "latest": latest}


apply_overrides(globals())

src = Path(SOURCE_DIR)
vcs = Git(
    branch_regex=BRANCH_REGEX,
    tag_regex=TAG_REGEX,
    buffer_size=1 * 10**9,
    predicate=file_predicate([src]),
)

creator = VenvWrapper(with_pip=True)
shared_env_kwargs = dict(
    temporary=SEQUENTIAL, creator=creator, venv=Path(VENV_DIR_NAME)
)
ENVIRONMENT = {
    None: DynamicPip.factory(**shared_env_kwargs, args=["-e", "."] + SPHINX_DEPS),
    "local": Environment.factory(),
}


async def selector(rev, keys):
    """Select the ENVIRONMENT entry for a revision: 'local' for local/mock builds, else the shared config."""
    if rev.name == "local":
        return "local"
    return None


CustomDriver(
    root,
    OUTPUT_DIR,
    vcs=vcs,
    builder=SphinxBuilder(
        src,
        args=SPHINX_ARGS.split(),
        pre_cmd=["python", str(root / src / "pre-build.py"), Placeholder.SOURCE_DIR],
    ),
    env=ENVIRONMENT,
    selector=selector,
    encoder=PyDataVersionEncoder(),
    data_factory=data,
    root_data_factory=root_data,
    template_dir=root / src / "polyversion" / "templates",
    mock=MOCK_DATA,
).run(MOCK, SEQUENTIAL)
