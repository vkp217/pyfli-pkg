"""Importers and plotting helpers for pre-processed FLIM analysis outputs (AlliG, BH, PyFli)."""
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import json

class DatasetPlotter:
    """Mixin providing a standard plotting routine for imported dataset maps.

    Expects the including class to expose a ``self.dataset`` dict with a
    ``dataset['results']['maps']`` mapping of map name to array, as built by
    :class:`AlliGprocessedImport` and :class:`BHprocessedImport`.
    """
    def _plotting_maps(self):
        if not hasattr(self, 'dataset') or not self.dataset or not self.dataset['results']['maps']:
            print("No maps found to plot.")
            return

        map_results = self.dataset['results']['maps']
        name_keys = list(map_results.keys())
        num_plots = len(name_keys)
        fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))
        
        if num_plots == 1: 
            axes = [axes]

        for i, key in enumerate(name_keys):
            data = map_results[key]
            
            if data.ndim == 2 and data.shape[1] == 2:
                g_vals, s_vals = data[:, 0], data[:, 1]
                h = axes[i].hist2d(g_vals, s_vals, bins=256, cmap='jet', range=[[0, 1], [0, 0.6]], cmin=1)
                xc = np.linspace(0, 1, 100)
                yc = np.sqrt(0.25 - (xc - 0.5)**2)
                axes[i].plot(xc, yc, color='black', linestyle='--', linewidth=1, alpha=0.7)
                axes[i].set_title(f"{self.dataset['name']} - Phasor")
                plt.colorbar(h[3], ax=axes[i], fraction=0.046, pad=0.04)
            else:
                if data.ndim == 1:
                    side = int(np.sqrt(len(data)))
                    data = data[:side*side].reshape((side, side))
                im = axes[i].imshow(data, cmap='magma')
                axes[i].set_title(f"{key}")
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()
        return fig

class AlliGprocessedImport(DatasetPlotter):
    """Imports pre-processed AlliG fit-result maps (``*Map.txt``) and ROI masks from a folder.

    Populates ``self.dataset['results']['maps']`` with mono- or bi-exponential
    fit parameter maps, and applies any ``*.roiN`` ROI mask files found in
    the same folder.
    """
    def __init__(self, folder_path):
        """Set up the dataset structure for a given AlliG output folder.

        Args:
            folder_path: Path to the folder containing AlliG ``*Map.txt``
                and ``*.roiN`` files.
        """
        self.folder_path = os.path.abspath(folder_path)
        self.dataset = {
            "name": os.path.basename(self.folder_path),
            "results": {
                "maps": {},
                "decay": None,
                "irf": None
            }
        }

    def _read_roi_files(self):
        """Processes .roiN files using the provided working logic."""
        if not os.path.exists(self.folder_path):
            print(f"Error: The path '{self.folder_path}' does not exist.")
            return None

        search_path = os.path.join(self.folder_path, "*.roiN")
        files = glob.glob(search_path)
        
        # Ensure we have a map to get dimensions from
        if not self.dataset['results']['maps']:
            print("Warning: No maps loaded yet. ROI mask cannot determine dimensions. Load text files first.")
            return

        # Get shape from the first available map
        first_map = list(self.dataset['results']['maps'].values())[0]
        a1, a2 = first_map.shape
        mask = np.zeros((a1, a2))

        for f_n in files:
            # --- YOUR WORKING CODE ---
            with open(f_n, 'r') as fid:
                j_data = json.load(fid)
            n = len(j_data['Named ROI Descriptions'])
            for i in range(n):
                a = j_data['Named ROI Descriptions'][i]["ROI Descriptor"]["Contours"][0]["Coordinates"]
                mask[a[1], a[0]] = 1
                self.dataset['results']['maps']['mask'] = mask
            # -------------------------

    def _detect_fit_type(self, files):
        for f in files:
            name_lower = os.path.basename(f).lower()
            if any(indicator in name_lower for indicator in ['tau_1', 'tau_2', 'f1_a', 'f1_i', 'a_1', 'a_2']):
                return True
        return False    

    def _read_text_files(self):
        if not os.path.exists(self.folder_path):
            print(f"Error: The path '{self.folder_path}' does not exist.")
            return None

        search_path = os.path.join(self.folder_path, "*Map.txt")
        files = glob.glob(search_path)
        
        if not files:
            print(f"No *Map.txt files found in {self.folder_path}.")
            return None

        is_biexp = self._detect_fit_type(files)

        for file_path in files:
            full_name = os.path.basename(file_path)
            name_no_ext = os.path.splitext(full_name)[0]
            parts = name_no_ext.split(' ')
            var_name = parts[-2]

            if is_biexp:
                biexp_rename = {'A_1 Map': "A1_map", 'A_2 Map':"A2_map", 'Baseline Map': 'baseline_map', 
                                'Chi^2 Map': 'chi2_map', 'f1_a Map':'f1a_map', 'f1_i Map':'f1i_map',
                                'f2_a Map':'f2a_map', 'f2_i Map':'f2i_map', 'Offset Map':'v_shift_map', 'R^2 Map': 'r2_map',
                                'tau_1 Map':'tau1_map', 'tau_2 Map':'tau2_map', '-tau-_a Map':'tau_mean_map','-tau-_i Map':'taui_mean_map'} 
                try:
                    key_lookup = var_name + " Map"
                    if key_lookup in biexp_rename:
                        data = np.loadtxt(file_path, delimiter='\t')
                        self.dataset['results']['maps'][biexp_rename[key_lookup]] = data
                except Exception as e:
                    print(f"Could not load {full_name}: {e}")
            else:
                monoexp_rename = {'A Map':'A_map', 'Baseline Map':'baseline_map', 
                                  'Chi^2 Map':'chi2_map', 'Offset Map':'v_shift_map',
                                  'R^2 Map':'r2_map', 'tau Map':'tau_map'}
                try:
                    key_lookup = var_name + " Map"
                    if key_lookup in monoexp_rename:
                        data = np.loadtxt(file_path, delimiter='\t')
                        self.dataset['results']['maps'][monoexp_rename[key_lookup]] = data
                except Exception as e:
                    print(f"Could not load {full_name}: {e}")

        # Automatically trigger ROI reading AFTER maps are established
        self._read_roi_files()
        
        # Summary Printout
        loaded_keys = list(self.dataset['results']['maps'].keys())
        print(f"--- AlliG Import Summary ---")
        print(f"Folder: {self.dataset['name']}")
        print(f"Total Maps Loaded: {len(loaded_keys)}")
        print(f"Map Names: {', '.join(loaded_keys)}")
        print(f"----------------------------")    

        return self.dataset


class BHprocessedImport(DatasetPlotter):
    """Imports pre-processed Becker & Hickl (BH) fit-result maps from ``.asc`` files.

    Populates ``self.dataset['results']['maps']`` with mono- or bi-exponential
    fit parameter maps, the phasor map, IRF, and the raw binned decay cube
    (from a ``binned_raw_data`` file), parsed from ``.asc`` files in the
    given folder.
    """
    def __init__(self, folder_path):
        """Set up the dataset structure for a given BH output folder.

        Args:
            folder_path: Path to the folder containing BH ``.asc`` files.
        """
        self.folder_path = os.path.abspath(folder_path)
        self.dataset = {
            "name": os.path.basename(self.folder_path),
            "results": {
                "maps": {},
                "decay": None,
                "irf": None
            }
        }

    def _detect_fit_type(self, files):
        for f in files:
            name_lower = os.path.basename(f).lower()
            if "biexp" in name_lower or any(x in name_lower for x in ["_a1", "_a2", "_t1", "_t2"]):
                return True
        return False

    def _read_ascfiles(self):
        if not os.path.exists(self.folder_path):
            print(f"Error: The path '{self.folder_path}' does not exist.")
            return None

        search_path = os.path.join(self.folder_path, "*.asc")
        files = glob.glob(search_path)
        
        if not files:
            print(f"No .asc files found in {self.folder_path}. Aborting import.")
            return None
        
        is_biexp = self._detect_fit_type(files)
        
        biexp_rename = {'a1': 'A1_map', 'a1[%]': 'A1_perc_map', 'a2': 'A2_map', 'a2[%]': 'A2_perc_map',
                        't1': 'tau1_map', 't2': 'tau2_map', 'tm': 'tau_mean_map', 'offset': 'v_shift_map', 
                        'chi': 'chi2_map', 'r2': 'r2_map', 'phasor': "phasor_map"}
        
        mono_rename = {'a': 'A_map', 'a[%]': 'A_perc_map', 't': 'tau_map', 
                       'offset': 'v_shift_map', 'chi': 'chi2_map', 'r2': 'r2_map', 'phasor': "phasor_map"}

        special_cases = ['phasor', 'irf'] 

        for file_path in files:
            full_name = os.path.basename(file_path)
            root_name, _ = os.path.splitext(full_name)
            name_no_ext = root_name.lower()            
            parts = name_no_ext.split('_')
            var_suffix = parts[-1] 
            
            try:
                if "binned_raw_data" in name_no_ext:
                    raw_cube = np.genfromtxt(file_path, skip_header=11, max_rows=512*512, encoding='latin1')
                    self.dataset["results"]["decay"] = raw_cube.reshape((512, 512, 256))
                    continue

                current_rename = biexp_rename if is_biexp else mono_rename
                map_names = ['a1','a1[%]','a2','a2[%]', 'chi', 'offset', 't1', 't2', 'a', 't']

                if var_suffix in map_names:
                    data = np.genfromtxt(file_path, skip_header=10, max_rows=512, encoding='latin1')
                    final_name = current_rename.get(var_suffix, var_suffix)
                    self.dataset["results"]["maps"][final_name] = data

                elif var_suffix in special_cases:
                    if var_suffix == 'phasor':
                        phasor_read = np.genfromtxt(file_path, invalid_raise=False)
                        self.dataset["results"]["maps"]["phasor_map"] = phasor_read
                    elif var_suffix == 'irf':
                        irf_read = np.genfromtxt(file_path, invalid_raise=False)
                        irf_1d = irf_read[:, 1] if irf_read.ndim == 2 else irf_read.flatten()
                        self.dataset["results"]["irf"] = np.tile(irf_1d, (512, 512, 1))

            except Exception as e:
                print(f"Skipping BH file {full_name} due to error: {e}")
                
        # Summary Printout
        loaded_keys = list(self.dataset['results']['maps'].keys())
        print(f"--- BH Import Summary ---")
        print(f"Folder: {self.dataset['name']}")
        print(f"Total Maps Loaded: {len(loaded_keys)}")
        print(f"Map Names: {', '.join(loaded_keys)}")
        print(f"-------------------------")
        return self.dataset

class PyFliprocessedImport(DatasetPlotter):
    """Placeholder importer for pyfli-native processed datasets.

    Reserved for a future importer of pyfli's own processed output format;
    currently defines no additional behavior beyond :class:`DatasetPlotter`.
    """
    pass