import numpy as np

class ROIoperations:
    def __init__(self):
        pass
    
    def extract_roi_datasets(self, global_dataset, multi_roi_mask, model_type='bi-exponential'):
        roi_datasets = {}
        H, W = multi_roi_mask.shape

        global_results = global_dataset.get('results', {})
        global_maps = global_results.get('maps', {})
        global_tr = global_results.get('TR_maps', {})
        T = global_tr['fit_map'].shape[2] if 'fit_map' in global_tr else 0

        roi_ids = np.unique(multi_roi_mask)
        roi_ids = roi_ids[roi_ids != 0]

        for rid in roi_ids:
            idx = (multi_roi_mask == rid)
            local_maps = {}
            for key, global_map_data in global_maps.items():
                local_map = np.zeros((H, W), dtype=np.float32)
                local_map[idx] = global_map_data[idx]
                local_maps[key] = local_map

            local_tr = {
                'fit_map': np.zeros((H, W, T), dtype=np.float32),
                'residual_map': np.zeros((H, W, T), dtype=np.float32)
            }
            
            if 'fit_map' in global_tr:
                local_tr['fit_map'][idx, :] = global_tr['fit_map'][idx, :]
            if 'residual_map' in global_tr:
                local_tr['residual_map'][idx, :] = global_tr['residual_map'][idx, :]

            # 3. Assemble the dataset structure
            roi_datasets[str(rid)] = {
                'name': f"ROI_Extraction_{rid}",
                'results': {
                    'maps': local_maps,
                    'TR_maps': local_tr
                }
            }

        return roi_datasets