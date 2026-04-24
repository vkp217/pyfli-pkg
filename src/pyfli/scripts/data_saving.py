import json
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class DataSaver:
    def __init__(self, path, folder_name="pyfli_Analysis_Results", new_session=True):
        #  base directory
        if os.path.isdir(path):
            base_dir = path
        else:
            base_dir = os.path.dirname(path)
        self.save_dir = os.path.join(base_dir, folder_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_file = os.path.join(self.save_dir, "pyfli_log.txt")        
        # only if starting a new session
        if new_session:
            self.log("\n" + "="*40)
            self.log(f"NEW SESSION STARTED")
            self.log("="*40)

    def log(self, message):
        """Appends a timestamped message to the log file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"
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
        self.log(f"JSON SAVED >> {name}.json")

    def save_npy(self, name, array):
        """Saves numpy arrays."""
        path = os.path.join(self.save_dir, f"{name}.npy")
        np.save(path, array)
        self.log(f"ARRAY SAVED >> {name}.npy | Shape: {array.shape}")

    def save_params(self, **kwargs):
        """Quickly logs multiple parameters."""
        for key, value in kwargs.items():
            self.log(f"PARAM >> {key}: {value}")