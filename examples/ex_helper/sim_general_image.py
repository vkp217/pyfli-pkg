from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Optional scipy import for outline erosion
try:
    from scipy import ndimage

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class BaseShape(ABC):
    """Abstract Base Class for ROI shape generation."""

    @abstractmethod
    def generate_masks(
        self, canvas_shape: tuple[int, int], box: tuple[int, int, int, int]
    ) -> list[np.ndarray]:
        """Returns a list of boolean 2D masks (one per ROI)."""


# ============================================================================
# CONCRETE SHAPE IMPLEMENTATIONS
# ============================================================================
class LettersShape(BaseShape):
    """Generates glyph/text ROIs scaled uniformly."""

    def __init__(
        self,
        letters: Sequence[str] = ("R", "P", "I"),
        gap: float = 0.15,
        font_path: str | None = None,
    ):
        self.letters = list(letters)
        self.gap = gap
        self.font_path = font_path or self._find_font()

    def _find_font(self) -> str:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/Library/Fonts/Arial.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        try:
            import matplotlib.font_manager as fm

            return fm.findfont(fm.FontProperties(weight="bold"))
        except Exception as exc:
            raise FileNotFoundError(
                "No TrueType font found. Pass font_path explicitly."
            ) from exc

    def generate_masks(
        self, canvas_shape: tuple[int, int], box: tuple[int, int, int, int]
    ) -> list[np.ndarray]:
        h, w = canvas_shape
        top, bottom, left, right = box
        avail_w, avail_h = w - left - right, h - top - bottom
        n = len(self.letters)

        if n == 0:
            raise ValueError("`letters` must contain at least one character.")

        slot_w = avail_w / n
        slot_w_inner = slot_w * (1.0 - self.gap)

        ref_font = ImageFont.truetype(self.font_path, 256)
        min_scale = float("inf")
        for letter in self.letters:
            x0, y0, x1, y1 = ref_font.getbbox(letter)
            gw, gh = max(x1 - x0, 1), max(y1 - y0, 1)
            min_scale = min(min_scale, min(slot_w_inner / gw, avail_h / gh))

        oversample = 3
        shared_size = max(1, round(256 * min_scale * oversample))
        font = ImageFont.truetype(self.font_path, shared_size)

        masks = []
        for i, letter in enumerate(self.letters):
            x0, y0, x1, y1 = font.getbbox(letter)
            gw, gh = max(x1 - x0, 1), max(y1 - y0, 1)

            img = Image.new("L", (gw, gh), 0)
            draw = ImageDraw.Draw(img)
            draw.text((-x0, -y0), letter, fill=255, font=font)

            arr = np.asarray(img)
            if oversample > 1:
                out_w = max(1, round(gw / oversample))
                out_h = max(1, round(gh / oversample))
                arr = np.asarray(
                    Image.fromarray(arr).resize((out_w, out_h), Image.LANCZOS)
                )

            glyph = arr > 127
            gh, gw = glyph.shape

            slot_x0 = left + i * slot_w
            cx = round(slot_x0 + (slot_w - gw) / 2.0)
            cy = round(top + (avail_h - gh) / 2.0)

            mask = np.zeros((h, w), dtype=bool)
            x0_c, y0_c = max(cx, 0), max(cy, 0)
            x1_c, y1_c = min(cx + gw, w), min(cy + gh, h)

            sub = glyph[y0_c - cy : y1_c - cy, x0_c - cx : x1_c - cx]
            mask[y0_c:y1_c, x0_c:x1_c] = sub
            masks.append(mask)

        return masks


class WellPlateShape(BaseShape):
    """Generates circular multiwell plate ROIs arranged in a grid."""

    def __init__(self, rows: int = 2, cols: int = 4, gap: float = 0.15):
        self.rows = rows
        self.cols = cols
        self.gap = gap

    def generate_masks(
        self, canvas_shape: tuple[int, int], box: tuple[int, int, int, int]
    ) -> list[np.ndarray]:
        h, w = canvas_shape
        top, bottom, left, right = box
        avail_w, avail_h = w - left - right, h - top - bottom

        slot_w, slot_h = avail_w / self.cols, avail_h / self.rows
        radius = (min(slot_w, slot_h) * (1.0 - self.gap)) / 2.0

        y_indices, x_indices = np.ogrid[:h, :w]
        masks = []

        for r in range(self.rows):
            for c in range(self.cols):
                cx = left + (c + 0.5) * slot_w
                cy = top + (r + 0.5) * slot_h
                dist_sq = (x_indices - cx) ** 2 + (y_indices - cy) ** 2
                masks.append(dist_sq <= radius**2)

        return masks


# ============================================================================
# MAIN ROI MASK GENERATOR & VISUALIZATION MANAGER
# ============================================================================
class ROIMaskGenerator:
    """ROI Generator, Previewer, and File Exporter."""

    def __init__(
        self,
        shape: tuple[int, int],
        *,
        top: int = 0,
        bottom: int = 0,
        left: int = 0,
        right: int = 0,
        background: int = 0,
        dtype=np.int32,
    ):
        self.shape = (int(shape[0]), int(shape[1]))
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.background = background
        self.dtype = dtype

        avail_w = self.shape[1] - self.left - self.right
        avail_h = self.shape[0] - self.top - self.bottom
        if avail_w <= 0 or avail_h <= 0:
            raise ValueError(
                f"Margins leave no room: avail=({avail_h},{avail_w}) for shape {self.shape}."
            )

    def generate_boolean_masks(
        self, shape_provider: BaseShape, outline_thickness: int = 0
    ) -> list[np.ndarray]:
        """Extract individual ROI boolean masks with optional outline thickness."""
        box = (self.top, self.bottom, self.left, self.right)
        raw_masks = shape_provider.generate_masks(self.shape, box)

        processed_masks = []
        for b_mask in raw_masks:
            if outline_thickness > 0 and HAS_SCIPY:
                eroded = ndimage.binary_erosion(b_mask, iterations=outline_thickness)
                processed_masks.append(b_mask & ~eroded)
            else:
                processed_masks.append(b_mask)
        return processed_masks

    # 1. False Color RGB Image Generator
    def generate_color_image(
        self,
        shape_provider: BaseShape,
        *,
        colors: list[tuple[int, int, int]] | None = None,
        outline_thickness: int = 0,
        seed: int = 0,
    ) -> np.ndarray:
        """Generates a 3-channel RGB uint8 false-color image."""
        masks = self.generate_boolean_masks(shape_provider, outline_thickness)
        n = len(masks)
        color_img = np.zeros((*self.shape, 3), dtype=np.uint8)

        if colors is None:
            rng = np.random.default_rng(seed)
            colors = [
                tuple(int(c) for c in rng.integers(60, 256, size=3)) for _ in range(n)
            ]
        elif len(colors) != n:
            raise ValueError(f"Colors count ({len(colors)}) != ROI count ({n}).")

        for b_mask, rgb in zip(masks, colors):
            color_img[b_mask] = rgb

        return color_img

    def generate_intensity_image(
        self,
        shape_provider: BaseShape,
        *,
        bit_depth: int = 8,
        intensities: list[float | int] | dict[int, float | int] | None = None,
        outline_thickness: int = 0,
        seed: int = 42,
    ) -> np.ndarray:
        """Generates a grayscale intensity image adhering to any custom `bit_depth` (1 to 64).

        Parameters
        ----------
        bit_depth : int
            Arbitrary bit depth (e.g., 10-bit, 12-bit, 14-bit, 32-bit).
            Values are scaled/clamped to [0, 2^bit_depth - 1] and stored
            in the smallest fitting NumPy uint container (uint8, uint16, uint32, or uint64).
        intensities : list or dict or None
            Per-ROI intensity values.
            - Floats in [0.0, 1.0] are scaled to max value (2^bit_depth - 1).
            - Integers are clamped to [0, 2^bit_depth - 1].
            - Dict maps ROI index -> intensity value.
            - None generates random intensities spread across [0.3*max, max].
        """
        if not (1 <= bit_depth <= 64):
            raise ValueError(f"`bit_depth` must be between 1 and 64, got {bit_depth}.")

        # 1. Dynamically assign the smallest fitting NumPy container type
        if bit_depth <= 8:
            target_dtype = np.uint8
        elif bit_depth <= 16:
            target_dtype = np.uint16
        elif bit_depth <= 32:
            target_dtype = np.uint32
        else:
            target_dtype = np.uint64

        # 2. Compute maximum intensity value for the given bit depth
        max_val = (1 << bit_depth) - 1

        masks = self.generate_boolean_masks(shape_provider, outline_thickness)
        n = len(masks)

        # 3. Resolve per-ROI values scaled to max_val
        if intensities is None:
            rng = np.random.default_rng(seed)
            low_bound = int(max_val * 0.3)
            raw_vals = [int(rng.integers(low_bound, max_val + 1)) for _ in range(n)]
        else:
            if isinstance(intensities, dict):
                intens_list = [intensities.get(i, 0.5) for i in range(n)]
            else:
                intens_list = list(intensities)

            if len(intens_list) != n:
                raise ValueError(
                    f"Intensities count ({len(intens_list)}) != ROI count ({n})."
                )

            raw_vals = []
            for v in intens_list:
                if isinstance(v, (float, np.floating)):
                    val = round(np.clip(v, 0.0, 1.0) * max_val)
                else:
                    val = int(np.clip(v, 0, max_val))
                raw_vals.append(val)

        # 4. Construct intensity array using the target container
        intensity_img = np.zeros(self.shape, dtype=target_dtype)
        for b_mask, val in zip(masks, raw_vals):
            intensity_img[b_mask] = val

        return intensity_img

    # 3. Binary Mask Generator ({0, 1} strictly)
    def generate_binary_mask(
        self, shape_provider: BaseShape, outline_thickness: int = 0
    ) -> np.ndarray:
        """Generates a binary uint8 mask strictly with values in {0, 1}."""
        masks = self.generate_boolean_masks(shape_provider, outline_thickness)
        binary_mask = np.zeros(self.shape, dtype=np.uint8)
        for b_mask in masks:
            binary_mask[b_mask] = 1  # Strictly 1 for ROI, 0 for background
        return binary_mask

    # 4. Multicluster Labeled Mask Generator
    def generate_cluster_mask(
        self,
        shape_provider: BaseShape,
        *,
        labels: list[int] | None = None,
        outline_thickness: int = 0,
    ) -> np.ndarray:
        """Generates an integer label mask (1, 2, 3... per ROI)."""
        masks = self.generate_boolean_masks(shape_provider, outline_thickness)
        n = len(masks)

        if labels is None:
            resolved_labels = list(range(1, n + 1))
        else:
            resolved_labels = list(labels)
            if len(resolved_labels) != n:
                raise ValueError(
                    f"Label count ({len(resolved_labels)}) != ROI count ({n})."
                )

        mask = np.full(self.shape, self.background, dtype=self.dtype)
        for b_mask, lbl in zip(masks, resolved_labels):
            mask[b_mask] = lbl
        return mask

    def plot_preview(
        self,
        shape_provider: BaseShape,
        *,
        bit_depth: int = 8,
        outline_thickness: int = 0,
        intensities: list[float | int] | dict[int, float | int] | None = None,
        labels: list[int] | None = None,
        figsize: tuple[float, float] = (14, 3.5),
        show: bool = True,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Displays a 1x4 Matplotlib subplot comparing all 4 output types."""
        color_img = self.generate_color_image(
            shape_provider, outline_thickness=outline_thickness
        )
        intensity_img = self.generate_intensity_image(
            shape_provider,
            bit_depth=bit_depth,
            intensities=intensities,
            outline_thickness=outline_thickness,
        )
        binary_mask = self.generate_binary_mask(
            shape_provider, outline_thickness=outline_thickness
        )
        cluster_mask = self.generate_cluster_mask(
            shape_provider, labels=labels, outline_thickness=outline_thickness
        )

        fig, axes = plt.subplots(1, 4, figsize=figsize)

        # 1. False Color
        axes[0].imshow(color_img)
        axes[0].set_title("1. False Color (RGB)")

        # 2. Grayscale Intensity
        im1 = axes[1].imshow(intensity_img, cmap="gray")
        axes[1].set_title(f"2. Intensity ({bit_depth}-bit)")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # 3. Binary Mask {0, 1}
        axes[2].imshow(binary_mask, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("3. Binary Mask {0, 1}")

        # 4. Multicluster Mask
        im3 = axes[3].imshow(cluster_mask, cmap="tab20")
        axes[3].set_title("4. Multicluster Mask")
        fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        if show:
            plt.show()

        return fig, axes

    def save_all_masks(
        self,
        shape_provider: BaseShape,
        output_dir: str | Path,
        prefix: str = "roi",
        *,
        fmt: str = "png",
        bit_depth: int = 8,
        outline_thickness: int = 0,
        intensities: list[float | int] | dict[int, float | int] | None = None,
        labels: list[int] | None = None,
        colors: list[tuple[int, int, int]] | None = None,
    ) -> dict[str, str]:
        """Generates and saves all 4 image formats:
        - False Color Image
        - Grayscale Intensity Image (8 or 16-bit)
        - Binary Mask Image (values in {0, 1})
        - Multicluster Mask Image
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        fmt_list = ["png", "tiff"] if fmt.lower() == "both" else [fmt.lower()]
        saved_files = {}

        # Generate all 4 image types
        color_img = self.generate_color_image(
            shape_provider, colors=colors, outline_thickness=outline_thickness
        )
        intensity_img = self.generate_intensity_image(
            shape_provider,
            bit_depth=bit_depth,
            intensities=intensities,
            outline_thickness=outline_thickness,
        )
        binary_mask = self.generate_binary_mask(
            shape_provider, outline_thickness=outline_thickness
        )
        cluster_mask = self.generate_cluster_mask(
            shape_provider, labels=labels, outline_thickness=outline_thickness
        )

        for ext in fmt_list:
            # 1. False Color Image (RGB uint8)
            col_file = out_path / f"{prefix}_color.{ext}"
            Image.fromarray(color_img).save(col_file)
            saved_files[f"color_{ext}"] = str(col_file)

            # 2. Grayscale Intensity Image (uint8 or uint16 according to bit depth)
            int_file = out_path / f"{prefix}_intensity_{bit_depth}bit.{ext}"
            Image.fromarray(intensity_img).save(int_file)
            saved_files[f"intensity_{ext}"] = str(int_file)

            # 3. Binary Mask Image (uint8 array with strictly {0, 1})
            bin_file = out_path / f"{prefix}_binary_mask.{ext}"
            Image.fromarray(binary_mask).save(bin_file)
            saved_files[f"binary_{ext}"] = str(bin_file)

            # 4. Multicluster Labeled Mask Image
            cluster_dtype = np.uint8 if cluster_mask.max() <= 255 else np.uint16
            cluster_file = out_path / f"{prefix}_multicluster_mask.{ext}"
            Image.fromarray(cluster_mask.astype(cluster_dtype)).save(cluster_file)
            saved_files[f"cluster_{ext}"] = str(cluster_file)

        return saved_files


if __name__ == "__main__":
    h, w = 512, 512
    generator = ROIMaskGenerator((h, w), top=24, bottom=24, left=24, right=24)

    well_shape = WellPlateShape(rows=2, cols=4, gap=0.15)

    custom_intensities = [
        0.1,
        0.25,
        0.4,
        0.55,
        0.7,
        0.8,
        0.9,
        1.0,
    ]  # custom intensities specified as float ratios per ROI

    # Preview
    generator.plot_preview(
        well_shape,
        bit_depth=10,
        intensities=custom_intensities,
        show=True,
    )

    # base_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    base_dir = "/mnt/e/Vikas/MyPyFli_temp/pf_website/simulation examples/wellplate_sim"
    saved_files = generator.save_all_masks(
        well_shape,
        output_dir=base_dir,
        prefix="well_plate_8",
        fmt="png",
        bit_depth=16,
        intensities=custom_intensities,
    )
