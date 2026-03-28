# solver/roi_maker.py 
import numpy as np
import cv2
import os
from skimage.segmentation import flood_fill

class ROIObject:
    def __init__(self, pts, roi_id):
        self.pts = np.array(pts, dtype=np.int32)
        self.roi_id = int(roi_id)  # Force integer ID
        self.center = np.mean(self.pts, axis=0)

    def move(self, dx, dy):
        self.pts += [int(dx), int(dy)]
        self.center += [int(dx), int(dy)]

    def rotate(self, angle_deg):
        rad = np.radians(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        M = np.array([[c, -s], [s, c]])
        relative_pts = self.pts - self.center
        self.pts = (relative_pts @ M.T + self.center).astype(np.int32)

    def scale(self, factor):
        relative_pts = self.pts - self.center
        self.pts = (relative_pts * factor + self.center).astype(np.int32)

class ROIMaker:
    def __init__(self, image_2d, save_path="masks/mask.npy"):
        self.raw_img = image_2d.astype(np.float64)
        self.display_base = cv2.normalize(image_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.H, self.W = image_2d.shape
        self.save_path = save_path
        
        self.rois = [] 
        self.selected_idx = -1
        self.current_roi_id = 1
        self.mode = 'rect' 
        self.drawing = False
        self.moving = False
        self.show_bg = True 
        self.last_mouse = (0, 0)
        self.pts = []
        self.preview_mask = None

        if os.path.exists(self.save_path):
            self.load_mask(self.save_path)

    def load_mask(self, path):
        try:
            loaded_mask = np.load(path)
            uids = np.unique(loaded_mask)
            for uid in uids[uids != 0]:
                contours, _ = cv2.findContours((loaded_mask == uid).astype(np.uint8), 
                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) > 5:
                        self.rois.append(ROIObject(cnt.reshape(-1, 2), uid))
            if len(self.rois) > 0:
                self.current_roi_id = max([r.roi_id for r in self.rois]) + 1
        except Exception as e:
            print(f"Load failed: {e}")

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = self._find_roi_at(x, y)
            if idx != -1:
                self.selected_idx = idx
                if flags & cv2.EVENT_FLAG_CTRLKEY: # Move mode
                    self.moving = True
                    self.last_mouse = (x, y)
            else:
                self.drawing = True
                self.pts = [(x, y)]
                self.selected_idx = -1

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.moving and self.selected_idx != -1:
                dx, dy = x - self.last_mouse[0], y - self.last_mouse[1]
                self.rois[self.selected_idx].move(dx, dy)
                self.last_mouse = (x, y)
            elif self.drawing and self.mode == 'freehand':
                self.pts.append((x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            if self.moving: self.moving = False
            if self.drawing:
                self.drawing = False
                new_pts = self._generate_pts(self.pts, x, y)
                if len(new_pts) > 2:
                    # Create new ROI with current ID
                    self.rois.append(ROIObject(new_pts, self.current_roi_id))
                    # AUTO-INCREMENT ID FOR NEXT ROI
                    self.current_roi_id += 1 
                    self.selected_idx = len(self.rois) - 1

    def _generate_pts(self, start_pts, x, y):
        ix, iy = start_pts[0]
        if self.mode == 'rect':
            return np.array([[ix, iy], [x, iy], [x, y], [ix, y]])
        elif self.mode == 'circle':
            r = int(np.sqrt((ix-x)**2 + (iy-y)**2))
            return cv2.ellipse2Poly((ix, iy), (r, r), 0, 0, 360, 10)
        return np.array(self.pts)

    def _find_roi_at(self, x, y):
        for i, roi in enumerate(self.rois):
            if cv2.pointPolygonTest(roi.pts, (x, y), False) >= 0: return i
        return -1

    def _update_display(self):
        canvas = cv2.cvtColor(self.display_base, cv2.COLOR_GRAY2BGR) if self.show_bg else np.zeros((self.H, self.W, 3), dtype=np.uint8)
        
        for i, roi in enumerate(self.rois):
            is_sel = (i == self.selected_idx)
            # Distinct colors based on ID
            color = [0, 255, 0] if is_sel else [(roi.roi_id*70)%255, (roi.roi_id*130)%255, 200]
            cv2.polylines(canvas, [roi.pts], True, color, 2 if is_sel else 1)
            
            # Transparency overlay
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [roi.pts], color)
            cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
            
            # Label
            cv2.putText(canvas, f"ID:{roi.roi_id}", tuple(roi.pts[0]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Real-time preview while drawing rect/circle
        if self.drawing and len(self.pts) > 0:
            cv2.polylines(canvas, [self._generate_pts(self.pts, self.last_mouse[0], self.last_mouse[1])], True, (255,255,255), 1)

        cv2.imshow('ROI Maker', canvas)

    def draw(self):
        win_name = 'ROI Maker'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, self._mouse_callback)
        
        while True:
            self._update_display()
            raw_key = cv2.waitKey(30) & 0xFF
            
            if raw_key == 27: break # ESC
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1: break

            key = chr(raw_key).lower() if raw_key < 256 else ""
            
            if key in ['r', 'c', 'f']: self.mode = {'r':'rect', 'c':'circle', 'f':'freehand'}[key]
            elif key == 'b': self.show_bg = not self.show_bg
            elif key == 'z' and self.selected_idx != -1:
                self.rois.pop(self.selected_idx)
                self.selected_idx = -1
                # Recalculate next ID so we don't have gaps or duplicates
                self.current_roi_id = (max([r.roi_id for r in self.rois]) + 1) if self.rois else 1
            elif key == 's' or raw_key == 13: # Save
                self.save_masks()
                break

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return self.get_multi_cluster_mask()

    def get_multi_cluster_mask(self):
        # Ensure int32 to hold IDs correctly
        mask = np.zeros((self.H, self.W), dtype=np.int32)
        for roi in self.rois:
            cv2.fillPoly(mask, [roi.pts.astype(np.int32)], int(roi.roi_id))
        return mask

    def save_masks(self):
        multi = self.get_multi_cluster_mask()
        np.save(self.save_path, multi)
        print(f"Saved multi-cluster mask with {len(np.unique(multi))-1} unique IDs.")

if __name__ == "__main__":
    from pyfli import DataOperations
    # from fligpuFitter import Fli_GPUProcessor 
    # from flicpuFitter import Fli_CPUProcessor
    # from globalFitter import GlobalFLIFitter

    # 1. Load Data
    loader = DataOperations(
        fli_path = "exp 14 MDT set1 24hrs/24hr/mouse L/mouseL_740BP_03",
        irf_path = "exp 14 MDT set1 24hrs/24hr/mouse L/mouseL_IRF_700nm_02"
    )
    fli_cube = loader.load_fli()
    irf_cube = loader.load_irf() # Assuming standard loader signature

    # 2. Draw Regions
    intensity_proj = np.sum(fli_cube, axis=2)
    maker = ROIMaker(intensity_proj, save_path="mouseL_mask.npy")
    binary_mask = maker.draw()
    multi = maker.get_multi_cluster_mask()
    import matplotlib.pyplot as plt
    plt.imshow(multi)
    plt.show()

    # # 3. Global Pipeline Execution
    # # Fli_CPUProcessor for cluster segmentation automatically intercepts multi-dimensional integer grids
    # freq = [80.0, 1000/(201*0.04)] # Freq values from your setup
    # cpu_fitter = Fli_CPUProcessor(freq=freq, fitter_class=GlobalFLIFitter)

    # results = cpu_fitter.process_image(
    #     image_cube=fli_cube,
    #     irf_cube=irf_cube,
    #     mask=multi_mask,
    #     estimator='global_mle', # or 'global_least_squares'
    #     model_type='bi-exponential'
    # )