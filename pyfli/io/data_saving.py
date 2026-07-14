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
        self, path: str, folder_name: str = "_pyfli_Analysis", new_session: bool = False
    ) -> None:
        #  base directory
        path = os.path.normpath(path)
        if os.path.isdir(path):
            base_name = os.path.basename(path)
            base_dir = path
        else:
            base_name = os.path.splitext(os.path.basename(path))[0]
            base_dir = os.path.dirname(path)
        self.save_dir = os.path.join(base_dir, base_name + folder_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_file = os.path.join(self.save_dir, base_name + "_pyfli_log.txt")
        # only if starting a new session
        if new_session:
            self.log("=" * 40)
            self.log(f"Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.log("=" * 40)

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
