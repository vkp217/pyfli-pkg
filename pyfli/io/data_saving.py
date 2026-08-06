"""
Save fitted maps, metadata, figures, and session artifacts to disk.

This module belongs to :mod:`pyfli.io` and is part of PyFLI detector importers, file
readers, saving helpers, and processed-data loaders. Public API includes classes
:class:`DataSaver`; functions :func:`filter_vars`.
"""

from typing import Any

from pyfli import logging

# pyfli/io/data_saving.py
import json
import os

import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt


def filter_vars(local_vars: np.ndarray, keys: np.ndarray) -> Any:
    """
    Run the filter vars routine.

    Parameters
    ----------
    local_vars : np.ndarray
        Local variable dictionary filtered before saving.
    keys : np.ndarray
        Dataset keys to include in the saved output.

    Returns
    -------
    Any
        Object produced by filter vars.
    """
    return {k: local_vars[k] for k in keys if k in local_vars}


class DataSaver:
    """
    Persist PyFLI fitting outputs and analysis artifacts. The class creates a session
    directory and writes arrays, metadata, figures, and tabular summaries in a
    consistent layout.

    Parameters
    ----------
    path : str
        Filesystem path loaded or saved by the routine.
    folder_name : str
        Output folder where session files are written.
    new_session : bool
        If ``True``, create a new timestamped session folder.
    """

    def __init__(
        self,
        path: str,
        folder_name: str | None = "_pyfli_Analysis",
        new_session: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        path : str
            Either:
              - a base file/dir path to derive a NEW session folder from
                (used together with `folder_name` to build the folder name), or
              - the exact path of an EXISTING session folder to reconnect to
                (used together with `folder_name=None`).
        folder_name : str | None
            - str  : suffix appended to `path`'s base name to build/create the
                     save directory, e.g. path="foo", folder_name="_Analysis"
                     -> save_dir=".../foo_Analysis" (created if missing).
            - None : `path` IS the save directory already -- connect to it
                     directly with no renaming/reconstruction. `path` must
                     already exist in this case.
        new_session : bool
            If True, writes a session-start banner to the log. Leave False
            when reconnecting to load existing data, so you don't append a
            spurious "session started" entry to an existing log.
        """
        path = os.path.normpath(path)

        if folder_name is None:
            # --- Reconnect to an existing session folder, as-is -----------
            if not os.path.isdir(path):
                raise FileNotFoundError(
                    f"folder_name=None expects an existing directory to "
                    f"connect to, but no such directory exists: {path}"
                )
            self.save_dir = path
            base_name = os.path.basename(self.save_dir)
        else:
            # --- Build a session folder from a base name -------------------
            self.save_dir, base_name = self._resolve_save_dir(path, folder_name)

        os.makedirs(self.save_dir, exist_ok=True)
        self.log_file = os.path.join(self.save_dir, base_name + "_pyfli_log.txt")

        # only if starting a new session
        if new_session:
            self.log("=" * 40)
            self.log(f"Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.log("=" * 40)

    @classmethod
    def load(cls, save_dir: str) -> "DataSaver":
        """
        Convenience constructor for reconnecting to an existing session
        folder for data loading. Equivalent to:
            DataSaver(path=save_dir, folder_name=None, new_session=False)
        """
        return cls(path=save_dir, folder_name=None, new_session=False)

    @staticmethod
    def _resolve_save_dir(path: str, folder_name: str) -> tuple[str, str]:
        """
        Pure path-arithmetic: given a base `path` and a `folder_name` suffix,
        returns (save_dir, base_name) -- the exact same save_dir that
        __init__ would create. No filesystem side effects (no makedirs),
        so this is safe to call just to find out where an auto-derived
        session folder WOULD be / already IS, without creating anything.

        This is the single source of truth for the "base name + suffix"
        folder-naming scheme, so save and load can never disagree about
        where an auto-derived folder lives.
        """
        path = os.path.normpath(path)
        if os.path.isdir(path):
            base_dir = os.path.dirname(path)
            base_name = os.path.basename(path)
        else:
            base_dir = os.path.dirname(path)
            base_name = os.path.splitext(os.path.basename(path))[0]
        save_dir = os.path.join(base_dir, base_name + folder_name)
        return save_dir, base_name

    @classmethod
    def resolve_path(cls, path: str, folder_name: str) -> str:
        """
        Public helper: returns the save_dir that DataSaver(path, folder_name)
        would resolve to, without creating it. Use this to derive the load
        path for an auto-saved session before deciding whether to load from
        or write into it.
        """
        save_dir, _ = cls._resolve_save_dir(path, folder_name)
        return save_dir

    def log(self, message: Any) -> None:
        """
        Run the log routine.

        Parameters
        ----------
        message : Any
            Message text displayed to the user.

        Returns
        -------
        None
            No object is returned; the function perform log.
        """
        formatted_msg = f"{message}"
        logging.info(formatted_msg)
        with open(self.log_file, "a") as f:
            f.write(formatted_msg + "\n")

    def log_to_file(self, message: Any) -> None:
        """
        Append to the log file only, without echoing through logging/console.

        Use for bulk/verbose content (e.g. a printed comparison table) that
        belongs in the session log but would be noisy to also emit via the
        logger on every line. For normal status messages, use ``log()``.

        Parameters
        ----------
        message : Any
            Message text appended to the log file.

        Returns
        -------
        None
            No object is returned; the message is written to ``self.log_file``.
        """
        with open(self.log_file, "a") as f:
            f.write(f"{message}\n")

    def save_plot(
        self, name: str, fig: Any | None = None, dpi: int = 300, close: bool = True
    ) -> None:
        """
        Save plot.

        Parameters
        ----------
        name : str
            Dataset, experiment, figure, or output name.
        fig : Any | None
            Matplotlib figure object to update or save.
        dpi : int
            Resolution used when saving a figure.
        close : bool
            Whether to close the figure after saving.

        Returns
        -------
        None
            No object is returned; the function save plot.
        """
        path = os.path.join(self.save_dir, f"{name}.png")
        target = fig if fig is not None else plt
        try:
            target.savefig(path, bbox_inches="tight", dpi=dpi)
            self.log(f"IMAGE SAVED >> {name}.png")
        except Exception as e:
            self.log(f"ERROR saving {name}: {str(e)}")

        if close:
            plt.close(fig) if fig else plt.close()

    def save_json(self, name: str, data_dict: np.ndarray) -> None:
        """Saves settings/dictionaries as JSON."""
        path = os.path.join(self.save_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data_dict, f, indent=4)
        self.log(f"JSON saved: >> {name}.json")

    def load_json(self, name: str) -> Any:
        """Loads a dictionary previously saved with save_json."""
        path = os.path.join(self.save_dir, f"{name}.json")
        with open(path, "r") as f:
            data = json.load(f)
        self.log(f"JSON loaded: << {name}.json")
        return data

    def save_npy(self, name: str, array: np.ndarray) -> None:
        """
        Save npy.

        Parameters
        ----------
        name : str
            Dataset, experiment, figure, or output name.
        array : np.ndarray
            Array processed by the routine.

        Returns
        -------
        None
            No object is returned; the function save npy.
        """
        path = os.path.join(self.save_dir, f"{name}.npy")
        np.save(path, array)
        if isinstance(array, dict):
            self.log(f"Array saved: >> {name}.npy | Type: Dictionary")
        elif isinstance(array, np.ndarray):
            self.log(f"Array saved: >> {name}.npy | Shape: {array.shape}")
        else:
            self.log(f"Array saved: >> {name}.npy")

    def load_npy(self, name: str, allow_pickle: bool = True) -> Any:
        """Loads an array/dict previously saved with save_npy."""
        path = os.path.join(self.save_dir, f"{name}.npy")
        array = np.load(path, allow_pickle=allow_pickle)
        # np.save on a dict wraps it in a 0-d object array; unwrap it back.
        if array.shape == () and array.dtype == object:
            array = array.item()
        if isinstance(array, dict):
            self.log(f"Array loaded: << {name}.npy | Type: Dictionary")
        elif isinstance(array, np.ndarray):
            self.log(f"Array loaded: << {name}.npy | Shape: {array.shape}")
        else:
            self.log(f"Array loaded: << {name}.npy")
        return array

    def save_params(self, **kwargs: Any) -> None:
        """
        Save params.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        None
            No object is returned; the function save params.
        """
        for key, value in kwargs.items():
            self.log(f"Parameter: >> {key}: {value}")

    def save_config(
        self, config_dict: np.ndarray, name: str = "fitting_config"
    ) -> None:
        """
        Save config.

        Parameters
        ----------
        config_dict : np.ndarray
            Configuration dictionary written to disk.
        name : str
            Dataset, experiment, figure, or output name.

        Returns
        -------
        None
            No object is returned; the function save config.
        """
        self.log(f"--- Configuration: {name} ---")
        serializable_config = {}
        for k, v in config_dict.items():
            self.log(f"SETTING >> {k}: {v}")
            if isinstance(v, (list, dict, str, int, float, bool, type(None))):
                serializable_config[k] = v
            else:
                serializable_config[k] = str(v)
        self.save_json(name, serializable_config)

    def load_config(self, name: str = "fitting_config") -> Any:
        """Loads a config dict previously saved with save_config."""
        config_dict = self.load_json(name)
        self.log(f"--- Configuration Loaded: {name} ---")
        for k, v in config_dict.items():
            self.log(f"SETTING << {k}: {v}")
        return config_dict
