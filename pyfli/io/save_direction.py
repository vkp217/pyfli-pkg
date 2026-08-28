"""
Turn a ``save_config`` dictionary into a ready :class:`DataSaver` session.

This module belongs to :mod:`pyfli.io` and is part of PyFLI detector importers, file
readers, saving helpers, and processed-data loaders. Public API includes the class
:class:`SaveLoadDirector`, which centralizes the branching that decides -- from a
single config dict -- whether a session is saved, loaded, or skipped, and where its
folder lives.
"""

# pyfli/io/save_direction.py
import os
from typing import Any, ClassVar

from pyfli import logging

from .data_saving import DataSaver


class SaveLoadDirector:
    """
    Resolve a ``save_config`` dictionary into a :class:`DataSaver` session.

    Instead of scattering ``if mode == ...`` / ``if auto_path ...`` checks through
    a script, collect every save/load decision in one dict and hand it to this
    class. :meth:`build` returns a writable session (``mode='save'``), reconnects
    to an existing one (``mode='load'``), or returns ``None`` (``mode=None``).

    Parameters
    ----------
    config : dict | None
        User-supplied settings. Any key that is omitted falls back to
        :attr:`DEFAULTS`. Recognized keys:

        ``mode`` : {'save', 'load', None}
            ``'save'`` builds a writable session, ``'load'`` reconnects to an
            existing one, ``None`` disables saving.
        ``auto_path`` : bool
            If ``True``, the session folder is derived from ``DATA_PATH``. If
            ``False``, a manual path is required (see below).
        ``DATA_PATH`` : str | None
            Source data path; required when ``auto_path=True`` (for both save
            and load).
        ``manual_load_path`` : str | None
            Exact existing session folder to load; used when ``mode='load'``
            and ``auto_path=False``.
        ``manual_save_root_path`` : str | None
            Root under which the session folder is created; required when
            ``mode='save'`` and ``auto_path=False`` (also accepted as a load
            root).
        ``bin_r`` : int
            Spatial binning radius; part of the derived ``save_suffix``.
        ``version`` : int
            Analysis version; part of the derived ``save_suffix``.
        ``new_session`` : bool
            Forwarded to :class:`DataSaver` when ``mode='save'``.
        ``MODEL_TYPE``, ``free_memory`` : carried through untouched for
            downstream consumers.
    **overrides : Any
        Convenience keyword overrides merged on top of ``config``.

    Attributes
    ----------
    config : dict
        The fully merged configuration, including the derived ``save_suffix``.
    """

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "MODEL_TYPE": "bi-exponential",
        "bin_r": 0,
        "free_memory": True,
        "mode": "save",  # 'save' | 'load' | None
        "auto_path": True,  # True -> path auto-derived from DATA_PATH
        "manual_load_path": None,
        "manual_save_root_path": None,
        "DATA_PATH": None,
        "new_session": True,
        "version": 1,
    }

    VALID_MODES: ClassVar[tuple[str | None, ...]] = ("save", "load", None)

    def __init__(self, config: dict | None = None, **overrides: Any) -> None:
        cfg = {**self.DEFAULTS, **(config or {}), **overrides}

        if cfg["mode"] not in self.VALID_MODES:
            raise ValueError(
                f"save_config['mode'] must be one of {self.VALID_MODES}, "
                f"got {cfg['mode']!r}"
            )

        cfg["save_suffix"] = f"bin{cfg['bin_r']}_pf_Analysis_v{cfg['version']}"
        self.config = cfg

    # ------------------------------------------------------------------
    # dict-like access, so existing ``cfg[...]`` call sites keep working
    # ------------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Return ``self.config[key]`` if present, else ``default``."""
        return self.config.get(key, default)

    def to_dict(self) -> dict:
        """Return a shallow copy of the merged configuration."""
        return dict(self.config)

    @property
    def save_suffix(self) -> str:
        """Folder-name suffix derived from ``bin_r`` and ``version``."""
        return self.config["save_suffix"]

    @property
    def mode(self) -> str | None:
        """Configured session mode: ``'save'``, ``'load'``, or ``None``."""
        return self.config["mode"]

    def __repr__(self) -> str:
        c = self.config
        return (
            f"{type(self).__name__}(mode={c['mode']!r}, "
            f"auto_path={c['auto_path']}, save_suffix={c['save_suffix']!r})"
        )

    # ------------------------------------------------------------------
    # path resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _data_saving_folder(data_path: str) -> str:
        """
        Base path that an auto-derived session hangs off of.

        A file path is used as-is; a directory path gets a trailing ``"_"``
        appended to its basename so :class:`DataSaver` builds a sibling folder
        rather than nesting inside the data directory.
        """
        if os.path.isfile(data_path):
            return data_path
        return os.path.join(
            os.path.dirname(data_path), f"{os.path.basename(data_path)}_"
        )

    def _auto_target(self) -> str:
        """Auto-derived base path, or raise if ``DATA_PATH`` is unset."""
        data_path = self.config["DATA_PATH"]
        if not data_path:
            raise ValueError(
                f"auto_path=True requires DATA_PATH to be set (mode={self.mode!r})."
            )
        return self._data_saving_folder(data_path)

    # ------------------------------------------------------------------
    # per-mode builders
    # ------------------------------------------------------------------
    def _build_load(self) -> DataSaver:
        cfg = self.config
        if cfg["auto_path"]:
            load_target = DataSaver.resolve_path(
                self._auto_target(), cfg["save_suffix"]
            )
            saver = DataSaver.load(load_target)
            logging.info(
                f'Data loaded (auto-derived path) - Session: "{saver.save_dir}"'
            )
            return saver

        if cfg["manual_load_path"]:
            load_target = cfg["manual_load_path"]
        elif cfg["manual_save_root_path"]:
            load_target = DataSaver.resolve_path(
                cfg["manual_save_root_path"], cfg["save_suffix"]
            )
        else:
            raise ValueError(
                "auto_path=False (mode='load') requires either 'manual_load_path' "
                "or 'manual_save_root_path' to be set."
            )
        saver = DataSaver.load(load_target)
        logging.info(f'Data loaded - Session: "{saver.save_dir}"')
        return saver

    def _build_save(self) -> DataSaver:
        cfg = self.config
        if cfg["auto_path"]:
            saver = DataSaver(
                path=self._auto_target(),
                folder_name=cfg["save_suffix"],
                new_session=cfg["new_session"],
            )
            logging.info(
                f'Data Saving (auto-derived path) - Session: "{saver.save_dir}"'
            )
            return saver

        if not cfg["manual_save_root_path"]:
            raise ValueError(
                "auto_path=False (mode='save') requires 'manual_save_root_path' to "
                "be set -- no path to save the data was given."
            )
        saver = DataSaver(
            path=cfg["manual_save_root_path"],
            folder_name=cfg["save_suffix"],
            new_session=cfg["new_session"],
        )
        logging.info(f'Data Saving - Session: "{saver.save_dir}"')
        return saver

    def _build_none(self) -> None:
        """``mode=None``: saver disabled entirely."""
        logging.info("No data will be saved in this session")
        return None

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------
    def build(self) -> DataSaver | None:
        """
        Construct the :class:`DataSaver` described by the configuration.

        Returns
        -------
        DataSaver | None
            A writable session (``mode='save'``), a reconnected session
            (``mode='load'``), or ``None`` (``mode=None``).
        """
        dispatch = {
            "save": self._build_save,
            "load": self._build_load,
            None: self._build_none,
        }
        return dispatch[self.config["mode"]]()

    @classmethod
    def from_config(
        cls, config: dict | None = None, **overrides: Any
    ) -> DataSaver | None:
        """
        One-shot helper: ``SaveLoadDirector(config, **overrides).build()``.

        Returns
        -------
        DataSaver | None
            Same as :meth:`build`.
        """
        return cls(config, **overrides).build()
