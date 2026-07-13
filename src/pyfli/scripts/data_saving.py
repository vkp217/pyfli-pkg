#  scripts/data_saving.py
"""Utilities for persisting FLI/FLIM analysis outputs to disk.

This module provides :class:`DataSaver`, a small helper that manages an
analysis output folder and writes plots, JSON, and NumPy arrays into it
while keeping a plain-text session log.
"""
import json
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def filter_vars(local_vars, keys):
    """Select a subset of entries from a variables dictionary.

    Args:
        local_vars: Dictionary to filter (e.g. the output of ``locals()``).
        keys: Iterable of keys to keep.

    Returns:
        dict: A new dictionary containing only the entries of `local_vars`
        whose key is present in `keys`.
    """
    return {k: local_vars[k] for k in keys if k in local_vars}

class DataSaver:
    """Manages a per-dataset analysis folder and session log.

    On construction, `DataSaver` resolves an output directory next to the
    given path (or inside it, if `path` is already a directory) and
    creates it if needed. It exposes helper methods for logging text,
    and saving plots, JSON, and NumPy arrays into that directory, with
    every action recorded to a text log file.

    Attributes:
        save_dir: Directory where outputs and the log file are written.
        log_file: Path to the plain-text session log file.
    """

    def __init__(self, path, folder_name="_pyfli_Analysis", new_session=False):
        """Resolve/create the analysis output directory and log file.

        Args:
            path: Path to a data file or an existing directory. The
                output folder is derived from this path's basename (with
                its extension stripped, if any) plus `folder_name`.
            folder_name: Suffix appended to the base name to form the
                output directory name. Defaults to "_pyfli_Analysis".
            new_session: If True, writes a session-start banner (with a
                timestamp) to the log file.
        """
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
            self.log("="*40)
            self.log(f"Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.log("="*40)

    def log(self, message):
        """Print a message and append it to the session log file.

        Args:
            message: Text to print and log.
        """
        formatted_msg = f"{message}"
        print(formatted_msg)
        with open(self.log_file, "a") as f:
            f.write(formatted_msg + "\n")

    def save_plot(self, name, fig=None, dpi=300, close=True):
        """Save a matplotlib figure as a PNG into the save directory.

        Args:
            name: Base filename (without extension) for the saved image.
            fig: Matplotlib figure to save. If None, the current pyplot
                figure (`plt`) is used instead.
            dpi: Resolution (dots per inch) for the saved image.
            close: If True, closes the figure (or the current pyplot
                figure) after saving.

        Note:
            Any exception raised while saving is caught and logged rather
            than propagated.
        """
        path = os.path.join(self.save_dir, f"{name}.png")
        target = fig if fig is not None else plt        
        try:
            target.savefig(path, bbox_inches='tight', dpi=dpi)
            self.log(f"IMAGE SAVED >> {name}.png")
        except Exception as e:
            self.log(f"ERROR saving {name}: {str(e)}")
        
        if close:
            plt.close(fig) if fig else plt.close()

    def save_json(self, name, data_dict):
        """Saves settings/dictionaries as JSON."""
        path = os.path.join(self.save_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data_dict, f, indent=4)
        self.log(f"JSON saved: >> {name}.json")

    def save_npy(self, name, array):
        """Save an array-like object as a `.npy` file in the save directory.

        Args:
            name: Base filename (without extension) for the saved file.
            array: Object to save with `numpy.save` (e.g. a NumPy array
                or a dictionary of arrays).
        """
        path = os.path.join(self.save_dir, f"{name}.npy")
        np.save(path, array)
        if isinstance(array, dict):
            self.log(f"Array saved: >> {name}.npy | Type: Dictionary")
        elif isinstance(array, np.ndarray):
            self.log(f"Array saved: >> {name}.npy | Shape: {array.shape}")
        else:
            self.log(f"Array saved: >> {name}.npy")
                     
    def save_params(self, **kwargs):
        """Log an arbitrary set of parameters as key/value pairs.

        Args:
            **kwargs: Parameter names and values to record to the log.
        """
        for key, value in kwargs.items():
            self.log(f"Parameter: >> {key}: {value}")

    def save_config(self, config_dict, name="fitting_config"):
        """Log a configuration dictionary and save it as JSON.

        Non-JSON-serializable values are converted to their string
        representation before being written out.

        Args:
            config_dict: Dictionary of configuration settings.
            name: Base filename (without extension) used both as the log
                section title and the saved JSON file name.
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