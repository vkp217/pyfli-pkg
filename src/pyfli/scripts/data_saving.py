import json
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def filter_vars(local_vars, keys):
    """Filters a dictionary of local variables by specific keys."""
    return {k: local_vars[k] for k in keys if k in local_vars}

class DataSaver:
    def __init__(self, path, folder_name="_pyfli_Analysis", new_session=False):
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
        """Appends a timestamped message to the log file."""
        formatted_msg = f"{message}"
        print(formatted_msg)
        # 'a' mode ensures we append to the existing log.txt
        with open(self.log_file, "a") as f:
            f.write(formatted_msg + "\n")

    def save_plot(self, name, fig=None, dpi=300, close=True):
        """
        Saves a plot. Handles subplots (pass fig) or direct plots (uses current).
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
        """Saves numpy arrays."""
        path = os.path.join(self.save_dir, f"{name}.npy")
        np.save(path, array)
        self.log(f"Array saved: >> {name}.npy | Shape: {array.shape}")

    def save_params(self, **kwargs):
        """Quickly logs multiple parameters."""
        for key, value in kwargs.items():
            self.log(f"Parameter: >> {key}: {value}")

    def save_config(self, config_dict, name="fitting_config"):
        """
        One-stop shop for saving script configurations.
        Logs every key-value pair and exports to JSON.
        """
        self.log(f"--- Configuration: {name} ---")
        
        # Clean the dictionary (convert non-serializable objects to strings)
        serializable_config = {}
        for k, v in config_dict.items():
            # Log it for the text file
            self.log(f"SETTING >> {k}: {v}")
            
            # Prepare for JSON (handle numpy arrays or classes)
            if isinstance(v, (list, dict, str, int, float, bool, type(None))):
                serializable_config[k] = v
            else:
                serializable_config[k] = str(v)
        
        # Save JSON
        self.save_json(name, serializable_config)