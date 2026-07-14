"""
flim_decay_cube.py
==================
Build a TCSPC decay-cube tensor from a Leica LIF file that contains
LifFlimImage data encoded with the patent-pending Leica "reduced Time Tagged"
compression scheme (US20230344447A1 / US12278654B2).

Dimensions explained
--------------------
M  – mosaic / tile index  (number of tiles or time-lapse frames)
Y  – image height (pixels)
X  – image width  (pixels)
H  – TCSPC histogram bin index  (number_bins_in_period ≈ laser-period / clock-period)

The resulting decay cube has shape (M, Y, X, H) with dtype uint16.
Each voxel [m, y, x, h] counts the photons that arrived in bin h of the
TCSPC histogram for pixel (y, x) of mosaic tile / frame m.

Compression format (from patent)
---------------------------------
The memory block is a standard zlib/Deflate stream (RFC 1950 / 1951).
Inside, photon data are stored as a sequence of 16-bit "reduced Time Tagged"
records, one record-stream per pixel, organised as follows (Fig. 2b / Fig. 5
of US20230344447A1):

  Record type         Bit layout (16 bits, little-endian)
  ----------------    ---------------------------------------------------
  Class marker        [bit15=0, bits14-11=detector(4), bit10=single_photon]
  Photon record       [bits6-0=arrival_time_low(7), bit7=has_extension]
  Extension record    [bits12-7=arrival_time_high(6)] – appended after
                        a photon record when bit7 of that record is set
  Pixel end marker    [bit15=1, bits9-0=run_length(skip empty pixels)]
  Line end marker     [specific reserved bit pattern, marks end of scan line]

Within each group (pixel), records are sorted by arrival time and then
delta-encoded: each record stores (current – previous) arrival time.
Decoding therefore requires a cumulative sum (prefix-sum) within each group.

Usage
-----
    from flim_decay_cube import build_decay_cube, plot_decay_cube
    import liffile

    with liffile.LifFile('example.lif') as lif:
        flim_img = lif.images['T23_005304_I1_2/FLIM']   # LifFlimImage
        cube = build_decay_cube(flim_img)                # (M, Y, X, H) uint16

    # Plot a single frame / pixel
    plot_decay_cube(cube, flim_img)

Notes
-----
* If you do NOT have the raw photon-tag data (i.e. the LifFlimImage memory
  block is unavailable or access is blocked), use `read_derived_images()`
  instead – it reads the pre-computed FLIM maps (intensity, decay time, etc.)
  that Leica already stores as ordinary LifImage objects in the same file.
* Tested against liffile >= 2026.4.11.
"""

from __future__ import annotations

import struct
import zlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import liffile as lf

# ---------------------------------------------------------------------------
# Part 1 – Decode the compressed photon-tag stream into a decay cube
# ---------------------------------------------------------------------------

# ── Record-type bit masks (16-bit little-endian words) ─────────────────────
# The patent describes a "reduced Time Tagged" 16-bit layout.
# Distinguish record types by their high bits:
#   bit15=1                       → pixel / line marker
#   bit15=0, bit14=0, bit13=0     → class marker   (starts a pixel group)
#   bit15=0, bit7=0               → photon record  (7-bit arrival time)
#   bit15=0, bit7=1               → photon record with extension

_MARKER_BIT      = 0x8000   # bit 15  – set for pixel/line markers
_LINE_END_MARKER = 0xFFFF   # sentinel used by some firmware versions
_CLASS_MARKER_ID = 0x0000   # upper nibble == 0  → class/detector marker

# Class marker layout:  [15:0] = [0, detector(4), single_photon(1), ...]
_CLASS_DET_SHIFT   = 11
_CLASS_DET_MASK    = 0x0F
_CLASS_SINGLE_BIT  = 0x0400   # bit 10

# Photon record layout: [15:0] = [0, ..., has_ext(1), arrival_low(7)]
_PHOT_ATIME_MASK   = 0x007F   # bits 6-0
_PHOT_EXT_BIT      = 0x0080   # bit 7  – if set, next record holds upper bits

# Extension record:     [15:0] = [0, ..., arrival_high(6), ...]
_EXT_ATIME_SHIFT   = 7        # upper 6 bits sit at bits 12-7
_EXT_ATIME_MASK    = 0x1F80

# Pixel-end marker:     [15:0] = [1, ..., run_length(10)]
_PIXEL_RUN_MASK    = 0x03FF   # bits 9-0


def _decompress_memory_block(flim_image: "lf.LifFlimImage") -> bytes:
    """
    Return the raw uint16 record stream from the LifFlimImage memory block.

    For Format='LMSRAW' the memory block is already uncompressed — it is a
    flat array of uint16 records written directly to disk.

    For older or alternate firmware that compresses the payload we fall back
    to a multi-layout zlib probe (offsets 0-256, chunked variants).
    """
    raw = flim_image.memory_block.read()
    fmt = flim_image.attrs["RawData"].get("Format", "")

    # LMSRAW: raw uint16 stream, no compression
    if fmt == "LMSRAW":
        return raw

    # Probe zlib at every 2-byte-aligned offset
    for offset in range(0, min(257, len(raw)), 2):
        for wbits in (15, -15, 31, 47):
            try:
                return zlib.decompress(raw[offset:], wbits=wbits)
            except zlib.error:
                pass

    # Chunked uint64 layout
    chunks: list[bytes] = []
    pos = 0
    while pos + 8 <= len(raw):
        sz = struct.unpack_from("<Q", raw, pos)[0]
        pos += 8
        if sz == 0 or pos + sz > len(raw):
            break
        for wbits in (15, -15, 31, 47):
            try:
                chunks.append(zlib.decompress(raw[pos: pos + sz], wbits=wbits))
                break
            except zlib.error:
                pass
        pos += sz
    if chunks:
        return b"".join(chunks)

    raise ValueError(
        f"Format={fmt!r}: could not decompress FLIM memory block "
        f"({len(raw):,} bytes).\nFirst 64 bytes: " + raw[:64].hex()
    )




def build_decay_cube(
    flim_image: "lf.LifFlimImage",
    *,
    channel: int = 0,
    dtype: np.dtype = np.uint16,
) -> np.ndarray:
    """
    Decode a LMSRAW PulseVersion2 photon-tag stream into a (M, Y, X, H) decay cube.

    Confirmed record layout (16-bit uint16 LE, uncompressed)
    ---------------------------------------------------------
    PHOTON record   — low_byte bit7 == 0  (low_byte < 0x80)
        bits[15:8]  arrival-time TCSPC bin  (0 .. H-1)
        bits[6:4]   detector channel        (3 bits)
        bits[3:0]   spare

    PIXEL CLOCK     — word == 0x54A0
        Each occurrence advances the current pixel index by 1.
        One pixel-clock per scanner pixel dwell period.

    SYNC A          — low_byte >= 0x80, word != 0x54A0, word != exact marker
        Laser-sync record that accompanies each pixel clock. Ignored.

    MARKER (exact)  — low_byte == 0xA0, high_byte in {1, 2, 4}
        0x01A0  LineStartMarker  — new scan line, reset pixel counter
        0x02A0  LineEndMarker    — end of scan line, advance line index
        0x04A0  FrameMarker      — new frame (when FrameRepetitionsMarked=True)

    Parameters
    ----------
    flim_image : LifFlimImage
    channel    : detector channel index (default 0)
    dtype      : accumulation dtype (use uint32 for very bright samples)

    Returns
    -------
    cube : np.ndarray, shape (M, Y, X, H)
    """
    sizes    = flim_image.sizes
    n_frames = sizes.get("M", 1)
    n_y      = sizes["Y"]
    n_x      = sizes["X"]
    n_bins   = sizes["H"]

    rd           = flim_image.attrs["RawData"]
    bidir        = bool(rd.get("BiDirectional",       False))
    invert_x     = bool(rd.get("InvertImageX",        False))
    invert_y     = bool(rd.get("InvertImageY",        False))
    line_start_m = int(rd.get("LineStartMarker",      1))
    line_end_m   = int(rd.get("LineEndMarker",        2))
    frame_m      = int(rd.get("FrameMarker",          4))

    # Exact 16-bit marker values  (low byte always 0xA0)
    MARKER_LINE_START = (line_start_m << 8) | 0xA0   # 0x01A0
    MARKER_LINE_END   = (line_end_m   << 8) | 0xA0   # 0x02A0
    MARKER_FRAME      = (frame_m      << 8) | 0xA0   # 0x04A0
    PIXEL_CLOCK       = 0x54A0                        # one per pixel dwell

    # ── Raw record stream (uncompressed uint16 LE) ─────────────────────────
    raw   = _decompress_memory_block(flim_image)
    recs  = np.frombuffer(raw[: (len(raw) // 2) * 2], dtype="<u2")

    # ── Output tensor ──────────────────────────────────────────────────────
    cube = np.zeros((n_frames, n_y, n_x, n_bins), dtype=dtype)

    # ── Scan state ─────────────────────────────────────────────────────────
    frame_idx = 0
    line_idx  = 0
    pixel_idx = -1   # -1 = before first pixel clock in this line
    in_line   = False

    for rec in recs:
        rec = int(rec)
        low  = rec & 0xFF
        high = rec >> 8

        # ── PHOTON: low byte bit7 == 0 ────────────────────────────────────
        if not (low & 0x80):
            if not in_line or pixel_idx < 0:
                continue                     # before first pixel clock
            atime = high
            ch    = (low >> 4) & 0x7
            if ch != channel or atime >= n_bins:
                continue

            # Bidirectional: odd lines are scanned right-to-left
            px = (n_x - 1 - pixel_idx) if (bidir and line_idx & 1) else pixel_idx
            py = (n_y - 1 - line_idx)  if invert_y                  else line_idx
            px = (n_x - 1 - px)        if invert_x                  else px

            if 0 <= px < n_x and 0 <= py < n_y and 0 <= frame_idx < n_frames:
                cube[frame_idx, py, px, atime] += 1
            continue

        # ── SYNC / MARKER: low byte bit7 == 1 ────────────────────────────
        if rec == MARKER_LINE_START:
            in_line   = True
            pixel_idx = -1          # reset; first PIXEL_CLOCK sets it to 0

        elif rec == MARKER_LINE_END:
            if in_line:
                line_idx += 1
                if line_idx >= n_y:
                    line_idx  = 0
                    frame_idx += 1
                    if frame_idx >= n_frames:
                        break
            in_line   = False
            pixel_idx = -1

        elif rec == MARKER_FRAME:
            frame_idx += 1
            line_idx   = 0
            pixel_idx  = -1
            in_line    = False
            if frame_idx >= n_frames:
                break

        elif rec == PIXEL_CLOCK:
            # Each 0x54A0 = one pixel dwell period elapsed → advance pixel
            if in_line:
                pixel_idx += 1
                if pixel_idx >= n_x:
                    pixel_idx = n_x - 1   # clamp; LINE_END will reset

        # else: SYNC_A laser record (0x04XX etc.) — ignored

    return cube



# ---------------------------------------------------------------------------
# Part 2 – Alternative: read pre-computed FLIM maps (no patent issues)
# ---------------------------------------------------------------------------

def read_derived_images(
    lif_file: "lf.LifFile",
    series_name: str,
) -> dict[str, np.ndarray]:
    """
    Read all Leica-computed FLIM parameter maps for a given image series.

    These are stored as ordinary LifImage objects (not LifFlimImage) so they
    are always accessible without touching the patent-protected raw stream.

    Parameters
    ----------
    lif_file : LifFile
        Open LifFile object.
    series_name : str
        Base name of the image series, e.g. ``'T23_005304_I1_2'``.

    Returns
    -------
    dict mapping short name → numpy array (shape M, Y, X per image).
    Keys include: 'Intensity', 'FastFlim', 'StdDev', 'PhasorReal',
    'PhasorImaginary', 'PhasorIntensity', 'PhasorMask',
    'DecayTime', 'Amplitude', 'TailOffset', 'IRFBackground',
    'IRFShift', 'FlimIntensity', 'AmplitudeSum', 'IntensitySum',
    'MeanPhotonArrivalTime', 'MeanDecayTime', 'ChiSquare'.
    """
    _name_map = {
        "Intensity":                              "Intensity",
        "Fast Flim":                              "FastFlim",
        "Standard Deviation":                     "StdDev",
        "Phasor Real":                            "PhasorReal",
        "Phasor Imaginary":                       "PhasorImaginary",
        "Phasor Intensity":                       "PhasorIntensity",
        "Phasor Mask":                            "PhasorMask",
        "FlimDecayTime 1 ch1":                    "DecayTime",
        "FlimAmplitude 1 ch1":                    "Amplitude",
        "FlimTailOffset 1 ch1":                   "TailOffset",
        "FlimInstrumentResponseFunctionBackground 1 ch1": "IRFBackground",
        "FlimInstrumentResponseFunctionShift 1 ch1":      "IRFShift",
        "FlimIntensity 1 ch1":                    "FlimIntensity",
        "FlimAmplitudeSum 1 ch1":                 "AmplitudeSum",
        "FlimIntensitySum 1 ch1":                 "IntensitySum",
        "FlimMeanPhotonArivalTime 1 ch1":          "MeanPhotonArrivalTime",
        "FlimMeanDecayTime 1 ch1":                "MeanDecayTime",
        "ChiSquare 1 ch1":                        "ChiSquare",
    }

    # Normalise: strip trailing /FLIM so 'T23_.../FLIM' and 'T23_...' both work
    base = series_name.rstrip("/")
    if base.endswith("/FLIM"):
        base = base[: -len("/FLIM")]

    # The derived images live at paths like:
    #   'T23_005304_I1_2/FLIM/Intensity'
    # img.name  = 'Intensity'              (only the last component)
    # img.path  = 'T23_005304_I1_2/FLIM/Intensity'  (full path)
    # We must match on img.path, not img.name.
    flim_prefix = f"{base}/FLIM/"

    result = {}
    for img in lif_file.images:
        if not img.path.startswith(flim_prefix):
            continue
        if img.is_flim:
            continue  # skip the LifFlimImage itself (raises NotImplementedError)
        # sub-image name = everything after 'Base/FLIM/'
        subname = img.path[len(flim_prefix):]
        key = _name_map.get(subname, subname)
        try:
            result[key] = img.asarray()
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Part 3 – Plotting helpers
# ---------------------------------------------------------------------------

def plot_decay_cube(
    cube: np.ndarray,
    flim_image: "lf.LifFlimImage | None" = None,
    *,
    frame: int = 0,
    pixel_yx: tuple[int, int] | None = None,
    save_path: str | None = None,
):
    """
    Visualise the decay cube with four panels:
      1. Intensity image  (sum over H axis)
      2. Mean arrival-time image  (weighted mean over H)
      3. Summed decay curve  (sum over all pixels in the frame)
      4. Single-pixel decay curve  (central pixel or specified pixel)

    Parameters
    ----------
    cube : np.ndarray, shape (M, Y, X, H)
    flim_image : LifFlimImage, optional
        Used to recover physical time axis (ns).
    frame : int
        Which mosaic tile / time frame to display.
    pixel_yx : (y, x) tuple, optional
        Pixel for the single-pixel decay panel.  Defaults to image centre.
    save_path : str, optional
        If given, save figure to this path instead of showing interactively.
    """
    import matplotlib.pyplot as plt

    M, Y, X, H = cube.shape
    frame = min(frame, M - 1)

    if pixel_yx is None:
        pixel_yx = (Y // 2, X // 2)
    py, px = pixel_yx

    # Time axis
    if flim_image is not None:
        t_ns = (np.arange(H) * flim_image.tcspc_resolution) * 1e9
        xlabel = "Arrival time (ns)"
    else:
        t_ns = np.arange(H)
        xlabel = "TCSPC bin"

    frame_cube = cube[frame]                      # (Y, X, H)
    intensity   = frame_cube.sum(axis=-1)         # (Y, X)

    # Weighted mean arrival time
    bins = t_ns if flim_image else np.arange(H)
    total = intensity.clip(min=1)
    mean_t = (frame_cube * bins[np.newaxis, np.newaxis, :]).sum(-1) / total

    summed_decay     = frame_cube.sum(axis=(0, 1))  # (H,)
    singlepix_decay  = frame_cube[py, px]           # (H,)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"FLIM Decay Cube  |  frame {frame}/{M-1}  |  shape {cube.shape}",
        fontsize=13,
    )

    # Panel 1 – Intensity
    im1 = axes[0, 0].imshow(intensity, cmap="hot", origin="upper")
    axes[0, 0].set_title("Intensity (photon count)")
    axes[0, 0].set_xlabel("X (pixels)"); axes[0, 0].set_ylabel("Y (pixels)")
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.85)

    # Panel 2 – Mean arrival time
    im2 = axes[0, 1].imshow(mean_t, cmap="RdYlGn_r", origin="upper")
    axes[0, 1].set_title("Mean photon arrival time")
    axes[0, 1].set_xlabel("X (pixels)"); axes[0, 1].set_ylabel("Y (pixels)")
    cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.85)
    cbar2.set_label(xlabel)
    # Mark the selected pixel
    axes[0, 1].plot(px, py, "c+", markersize=12, markeredgewidth=2,
                    label=f"pixel ({py},{px})")
    axes[0, 1].legend(fontsize=8)

    # Panel 3 – Summed decay (log-scale)
    axes[1, 0].semilogy(t_ns, summed_decay + 1, color="steelblue", linewidth=1.5)
    axes[1, 0].set_title("Summed decay (all pixels)")
    axes[1, 0].set_xlabel(xlabel); axes[1, 0].set_ylabel("Photon count")
    axes[1, 0].grid(True, which="both", alpha=0.3)

    # Panel 4 – Single pixel decay
    axes[1, 1].bar(t_ns, singlepix_decay, width=(t_ns[1] - t_ns[0]) if len(t_ns) > 1 else 1,
                   color="salmon", alpha=0.8)
    axes[1, 1].set_title(f"Single-pixel decay  (y={py}, x={px})")
    axes[1, 1].set_xlabel(xlabel); axes[1, 1].set_ylabel("Photon count")
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    return fig


def plot_derived_images(
    derived: dict[str, np.ndarray],
    series_name: str = "",
    frame: int = 0,
    save_path: str | None = None,
):
    """
    Display a grid of all pre-computed FLIM parameter maps.

    Parameters
    ----------
    derived : dict
        Output of :func:`read_derived_images`.
    series_name : str
        Title label.
    frame : int
        Which frame / mosaic tile to show for 3-D arrays.
    save_path : str, optional
        Save figure instead of displaying.
    """
    import matplotlib.pyplot as plt

    keys = list(derived.keys())
    n    = len(keys)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.suptitle(f"FLIM derived images – {series_name}  (frame {frame})", fontsize=13)
    axes = axes.flatten()

    for ax, key in zip(axes, keys):
        arr = derived[key]
        if arr.ndim == 3:
            data = arr[min(frame, arr.shape[0] - 1)]
        else:
            data = arr
        im = ax.imshow(data, cmap="viridis", origin="upper")
        ax.set_title(key, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Part 4 – Convenience wrapper: try raw cube first, fall back to derived maps
# ---------------------------------------------------------------------------

def load_flim_data(
    lif_path: str,
    series_name: str | None = None,
    *,
    channel: int = 0,
    use_derived_fallback: bool = True,
) -> tuple[np.ndarray | None, dict[str, np.ndarray], "lf.LifFlimImage | None"]:
    """
    High-level entry point.

    1. Opens the LIF file.
    2. Locates the LifFlimImage for *series_name* (or the first one found).
    3. Attempts to decode the raw photon-tag stream into a (M,Y,X,H) cube.
    4. If decoding fails (format not yet supported), reads the pre-computed
       derived FLIM maps instead.

    Returns
    -------
    cube : np.ndarray or None
        Shape (M, Y, X, H) if successfully decoded, else None.
    derived : dict
        Pre-computed FLIM parameter maps (always populated).
    flim_img : LifFlimImage or None
    """
    import liffile

    with liffile.LifFile(lif_path) as lif:
        # Find FLIM image.
        # img.name  = last path component, e.g. 'FLIM'
        # img.path  = full path,           e.g. 'T23_005304_I1_2/FLIM'
        # series_name may be supplied as either the full path ('T23_.../FLIM')
        # or just the base series name ('T23_...'), so we normalise both sides.
        def _flim_path_matches(img: "lf.LifImageABC") -> bool:
            if series_name is None:
                return True
            # Strip trailing /FLIM from the query so both forms work
            query = series_name.rstrip("/")
            if query.endswith("/FLIM"):
                query = query[: -len("/FLIM")]
            # img.path has the full path; strip /FLIM from it for comparison
            img_base = img.path.rstrip("/")
            if img_base.endswith("/FLIM"):
                img_base = img_base[: -len("/FLIM")]
            return img_base == query or img.path == series_name

        flim_img = None
        for img in lif.images:
            if img.is_flim and _flim_path_matches(img):
                flim_img = img
                break

        if flim_img is None:
            available = [img.path for img in lif.images if img.is_flim]
            raise ValueError(
                f"No LifFlimImage found for series_name={series_name!r}.\n"
                f"Available FLIM paths: {available}"
            )

        # Derive the series name from the FLIM image path (strip '/FLIM')
        base_series = flim_img.path.replace("/FLIM", "").lstrip("/")
        base_series = base_series.split("/")[0]

        # Read derived images (always works)
        derived = {}
        if use_derived_fallback:
            derived = read_derived_images(lif, base_series)

        # Attempt raw cube decode
        cube = None
        try:
            cube = build_decay_cube(flim_img, channel=channel)
            print(
                f"[OK] Decoded decay cube: shape={cube.shape}, "
                f"total photons={cube.sum():,}"
            )
        except (RuntimeError, ValueError, NotImplementedError) as exc:
            raw_hex = flim_img.memory_block.read()[:64].hex()
            print(f"[WARN] Could not decode raw TCSPC stream: {exc}")
            print(f"       Memory block first 64 bytes: {raw_hex}")
            print("       Using pre-computed derived images instead.")
            print(f"       Derived keys available: {list(derived.keys())}")

        return cube, derived, flim_img


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Part 5 – (X, Y, T) collapse and pixel-wise decay plotting
# ---------------------------------------------------------------------------

def collapse_to_xyt(cube: np.ndarray) -> np.ndarray:
    """
    Sum the M (mosaic/frame) axis to get a single (Y, X, H) image.

    Parameters
    ----------
    cube : np.ndarray, shape (M, Y, X, H)

    Returns
    -------
    xyt : np.ndarray, shape (Y, X, H)  — dtype promoted to uint32 to avoid overflow
    """
    return cube.sum(axis=0).astype(np.uint32)


def plot_xyt(
    xyt: np.ndarray,
    tcspc_resolution_s: float = 97e-12,
    *,
    pixel_yx: tuple[int, int] | None = None,
    cmap_intensity: str = "hot",
    cmap_lifetime:  str = "RdYlGn_r",
    save_path: str | None = None,
):
    """
    Interactive four-panel figure for a (Y, X, H) decay cube.

    Panels
    ------
    1. Intensity image          — total photon count per pixel
    2. Mean arrival-time image  — intensity-weighted mean TCSPC bin (ns)
    3. Summed decay curve       — log-scale, all pixels summed
    4. Single-pixel decay curve — bar chart for selected pixel

    Click anywhere on panels 1 or 2 to update the single-pixel decay.

    Parameters
    ----------
    xyt               : np.ndarray (Y, X, H)
    tcspc_resolution_s: bin width in seconds (default 97 ps)
    pixel_yx          : initial (y, x) selection; defaults to image centre
    cmap_intensity    : matplotlib colormap name for intensity image
    cmap_lifetime     : matplotlib colormap name for mean-arrival-time image
    save_path         : if given, save PNG instead of showing interactively
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_y, n_x, n_h = xyt.shape
    t_ns = np.arange(n_h) * tcspc_resolution_s * 1e9   # time axis in ns

    if pixel_yx is None:
        pixel_yx = (n_y // 2, n_x // 2)
    sel = list(pixel_yx)   # mutable so click handler can update it

    intensity = xyt.sum(axis=-1).astype(float)
    denom     = intensity.clip(min=1)
    mean_t    = (xyt * t_ns[np.newaxis, np.newaxis, :]).sum(-1) / denom

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"FLIM (Y={n_y}, X={n_x}, H={n_h})  |  "
        f"res={tcspc_resolution_s*1e12:.0f} ps/bin  |  "
        f"total photons={int(intensity.sum()):,}",
        fontsize=12,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    ax_int  = fig.add_subplot(gs[0, 0])
    ax_tau  = fig.add_subplot(gs[0, 1])
    ax_sum  = fig.add_subplot(gs[1, 0])
    ax_pix  = fig.add_subplot(gs[1, 1])

    # ── Panel 1: intensity ────────────────────────────────────────────────
    im1 = ax_int.imshow(intensity, cmap=cmap_intensity, origin="upper")
    ax_int.set_title("Intensity (photon count)")
    ax_int.set_xlabel("X (pixels)"); ax_int.set_ylabel("Y (pixels)")
    plt.colorbar(im1, ax=ax_int, shrink=0.85)
    marker_int, = ax_int.plot(*sel[::-1], "c+", ms=12, mew=2)

    # ── Panel 2: mean arrival time ────────────────────────────────────────
    vmax_t = t_ns[-1]
    im2 = ax_tau.imshow(mean_t, cmap=cmap_lifetime, origin="upper",
                        vmin=0, vmax=vmax_t)
    ax_tau.set_title("Mean photon arrival time (ns)")
    ax_tau.set_xlabel("X (pixels)"); ax_tau.set_ylabel("Y (pixels)")
    cb2 = plt.colorbar(im2, ax=ax_tau, shrink=0.85)
    cb2.set_label("ns")
    marker_tau, = ax_tau.plot(*sel[::-1], "w+", ms=12, mew=2)

    # ── Panel 3: summed decay ─────────────────────────────────────────────
    summed = xyt.sum(axis=(0, 1))
    ax_sum.semilogy(t_ns, summed + 1, color="steelblue", lw=1.5)
    ax_sum.set_title("Summed decay (all pixels, log scale)")
    ax_sum.set_xlabel("Arrival time (ns)"); ax_sum.set_ylabel("Photon count")
    ax_sum.grid(True, which="both", alpha=0.3)
    ax_sum.set_xlim(t_ns[0], t_ns[-1])

    # ── Panel 4: single-pixel decay ───────────────────────────────────────
    bw = t_ns[1] - t_ns[0] if n_h > 1 else 1.0
    bars = ax_pix.bar(t_ns, xyt[sel[0], sel[1]], width=bw,
                      color="salmon", alpha=0.85)
    ax_pix.set_xlabel("Arrival time (ns)"); ax_pix.set_ylabel("Photon count")
    pix_title = ax_pix.set_title(f"Pixel (y={sel[0]}, x={sel[1]})")
    ax_pix.grid(True, axis="y", alpha=0.3)
    ax_pix.set_xlim(t_ns[0], t_ns[-1])

    def _update_pixel(py: int, px: int) -> None:
        sel[0], sel[1] = int(py), int(px)
        decay = xyt[sel[0], sel[1]]
        for bar, h in zip(bars, decay):
            bar.set_height(h)
        ax_pix.set_ylim(0, max(decay.max() * 1.1, 1))
        pix_title.set_text(f"Pixel (y={sel[0]}, x={sel[1]})")
        for m in (marker_int, marker_tau):
            m.set_data([sel[1]], [sel[0]])
        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes in (ax_int, ax_tau) and event.xdata is not None:
            _update_pixel(int(np.clip(event.ydata, 0, n_y-1)),
                          int(np.clip(event.xdata, 0, n_x-1)))

    fig.canvas.mpl_connect("button_press_event", _on_click)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Part 5 – (X, Y, T) collapse and pixel-wise decay plotting
# ---------------------------------------------------------------------------

def collapse_to_xyt(cube: np.ndarray) -> np.ndarray:
    """
    Sum the M (mosaic/frame) axis to get a single (Y, X, H) image.

    Parameters
    ----------
    cube : np.ndarray, shape (M, Y, X, H)

    Returns
    -------
    xyt : np.ndarray, shape (Y, X, H)  — dtype promoted to uint32 to avoid overflow
    """
    return cube.sum(axis=0).astype(np.uint32)


def plot_xyt(
    xyt: np.ndarray,
    tcspc_resolution_s: float = 97e-12,
    *,
    pixel_yx: tuple[int, int] | None = None,
    cmap_intensity: str = "hot",
    cmap_lifetime:  str = "RdYlGn_r",
    save_path: str | None = None,
):
    """
    Interactive four-panel figure for a (Y, X, H) decay cube.

    Panels
    ------
    1. Intensity image          — total photon count per pixel
    2. Mean arrival-time image  — intensity-weighted mean TCSPC bin (ns)
    3. Summed decay curve       — log-scale, all pixels summed
    4. Single-pixel decay curve — bar chart for selected pixel

    Click anywhere on panels 1 or 2 to update the single-pixel decay.

    Parameters
    ----------
    xyt               : np.ndarray (Y, X, H)
    tcspc_resolution_s: bin width in seconds (default 97 ps)
    pixel_yx          : initial (y, x) selection; defaults to image centre
    cmap_intensity    : matplotlib colormap name for intensity image
    cmap_lifetime     : matplotlib colormap name for mean-arrival-time image
    save_path         : if given, save PNG instead of showing interactively
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_y, n_x, n_h = xyt.shape
    t_ns = np.arange(n_h) * tcspc_resolution_s * 1e9   # time axis in ns

    if pixel_yx is None:
        pixel_yx = (n_y // 2, n_x // 2)
    sel = list(pixel_yx)   # mutable so click handler can update it

    intensity = xyt.sum(axis=-1).astype(float)
    denom     = intensity.clip(min=1)
    mean_t    = (xyt * t_ns[np.newaxis, np.newaxis, :]).sum(-1) / denom

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"FLIM (Y={n_y}, X={n_x}, H={n_h})  |  "
        f"res={tcspc_resolution_s*1e12:.0f} ps/bin  |  "
        f"total photons={int(intensity.sum()):,}",
        fontsize=12,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    ax_int  = fig.add_subplot(gs[0, 0])
    ax_tau  = fig.add_subplot(gs[0, 1])
    ax_sum  = fig.add_subplot(gs[1, 0])
    ax_pix  = fig.add_subplot(gs[1, 1])

    # ── Panel 1: intensity ────────────────────────────────────────────────
    im1 = ax_int.imshow(intensity, cmap=cmap_intensity, origin="upper")
    ax_int.set_title("Intensity (photon count)")
    ax_int.set_xlabel("X (pixels)"); ax_int.set_ylabel("Y (pixels)")
    plt.colorbar(im1, ax=ax_int, shrink=0.85)
    marker_int, = ax_int.plot(*sel[::-1], "c+", ms=12, mew=2)

    # ── Panel 2: mean arrival time ────────────────────────────────────────
    vmax_t = t_ns[-1]
    im2 = ax_tau.imshow(mean_t, cmap=cmap_lifetime, origin="upper",
                        vmin=0, vmax=vmax_t)
    ax_tau.set_title("Mean photon arrival time (ns)")
    ax_tau.set_xlabel("X (pixels)"); ax_tau.set_ylabel("Y (pixels)")
    cb2 = plt.colorbar(im2, ax=ax_tau, shrink=0.85)
    cb2.set_label("ns")
    marker_tau, = ax_tau.plot(*sel[::-1], "w+", ms=12, mew=2)

    # ── Panel 3: summed decay ─────────────────────────────────────────────
    summed = xyt.sum(axis=(0, 1))
    ax_sum.semilogy(t_ns, summed + 1, color="steelblue", lw=1.5)
    ax_sum.set_title("Summed decay (all pixels, log scale)")
    ax_sum.set_xlabel("Arrival time (ns)"); ax_sum.set_ylabel("Photon count")
    ax_sum.grid(True, which="both", alpha=0.3)
    ax_sum.set_xlim(t_ns[0], t_ns[-1])

    # ── Panel 4: single-pixel decay ───────────────────────────────────────
    bw = t_ns[1] - t_ns[0] if n_h > 1 else 1.0
    bars = ax_pix.bar(t_ns, xyt[sel[0], sel[1]], width=bw,
                      color="salmon", alpha=0.85)
    ax_pix.set_xlabel("Arrival time (ns)"); ax_pix.set_ylabel("Photon count")
    pix_title = ax_pix.set_title(f"Pixel (y={sel[0]}, x={sel[1]})")
    ax_pix.grid(True, axis="y", alpha=0.3)
    ax_pix.set_xlim(t_ns[0], t_ns[-1])

    def _update_pixel(py: int, px: int) -> None:
        sel[0], sel[1] = int(py), int(px)
        decay = xyt[sel[0], sel[1]]
        for bar, h in zip(bars, decay):
            bar.set_height(h)
        ax_pix.set_ylim(0, max(decay.max() * 1.1, 1))
        pix_title.set_text(f"Pixel (y={sel[0]}, x={sel[1]})")
        for m in (marker_int, marker_tau):
            m.set_data([sel[1]], [sel[0]])
        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes in (ax_int, ax_tau) and event.xdata is not None:
            _update_pixel(int(np.clip(event.ydata, 0, n_y-1)),
                          int(np.clip(event.xdata, 0, n_x-1)))

    fig.canvas.mpl_connect("button_press_event", _on_click)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


# CLI demo  (python flim_decay_cube.py example.lif  [series_name])
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    lif_path    = sys.argv[1]
    series_name = sys.argv[2] if len(sys.argv) > 2 else None

    cube, derived, flim_img = load_flim_data(lif_path, series_name)

    if cube is not None:
        print(f"\nDecay cube shape : {cube.shape}  (M, Y, X, H)")
        print(f"  M = {cube.shape[0]}  mosaic tiles / time frames")
        print(f"  Y = {cube.shape[1]}  image height (pixels)")
        print(f"  X = {cube.shape[2]}  image width  (pixels)")
        print(f"  H = {cube.shape[3]}  TCSPC histogram bins")
        print(f"Total photon count : {cube.sum():,}")
        plot_decay_cube(cube, flim_img)
    elif derived:
        print(f"\nDerived FLIM maps available: {list(derived.keys())}")
        for k, v in derived.items():
            print(f"  {k:30s}  shape={v.shape}  dtype={v.dtype}")
        plot_derived_images(derived, series_name or "")
