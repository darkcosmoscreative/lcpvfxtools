import os
import argparse
import numpy as np
import OpenEXR
import sys
import cv2

print('lcpvfxtools.hdr_utils initialised')

# initialise config from module on path
current_dir = os.path.dirname(__file__)
module_root = os.path.abspath(os.path.join(current_dir, '../..'))
if module_root not in sys.path:
    sys.path.append(module_root)

from lcpvfxtools import config as cfg


def compute_weights(image, weighting='tent', min_val=0.001, max_val=2.36):
    """
    Compute per-pixel weights for HDR merge.
    - image: numpy array (H x W x C), assumed to be linear [0, ∞)
    - weighting: 'uniform', 'tent', or 'debevec'
    Returns: weight map (H x W)
    """
    gray = image.mean(axis=2)  # Approximate luminance

    if weighting == 'uniform':
        return np.ones_like(gray)

    elif weighting == 'tent':
        # Normalize to 0-1 range first
        norm = np.clip((gray - min_val) / (max_val - min_val), 0.0, 1.0)
        return 1.0 - np.abs(norm - 0.5) * 2.0

    elif weighting == 'debevec':
        # Apply Debevec triangle weight: w(z) = z - z_min if z <= z_mid, else z_max - z
        z_min, z_max = min_val, max_val
        z_mid = (z_min + z_max) / 2.0

        weights = np.where(
            gray <= z_mid,
            gray - z_min,
            z_max - gray
        )
        weights = np.clip(weights, 0.0, None)  # clamp negatives
        return weights

    else:
        raise ValueError(f"Unsupported weighting method: {weighting}")

def merge_hdr_linear(images, times, weighting='tent', min_val=0.001, max_val=2.36):
    """
    Merge a set of bracketed linear images into one HDR image.

    Parameters:
    - images: list of (H x W x C) NumPy arrays in float32/float64 linear space
    - times: list of exposure times in seconds (same length as images)
    - weighting: weighting method: 'uniform', 'tent', or 'debevec'
    - min_val, max_val: dynamic range for weighting functions

    Returns:
    - HDR image (H x W x C) in float32, linear, unclamped
    """
    assert len(images) == len(times), "Mismatched image/time count"

    images = [np.asarray(img, dtype=np.float32) for img in images]
    times = np.asarray(times, dtype=np.float32)
    log_times = np.log(times + 1e-8)

    h, w, c = images[0].shape
    hdr = np.zeros((h, w, c), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    for img, log_t in zip(images, log_times):
        img = np.clip(img, 0, 3) ** (1/2.2)  # Convert from sRGB to linear space
        w_map = compute_weights(img, weighting, min_val, max_val)
        w_map_exp = w_map[..., None]  # (H, W, 1) for broadcasting

        # Contribution to radiance estimate: w_ij * (I_ij / t_j)
        img = img ** 2.2
        hdr += w_map_exp * img / np.exp(log_t)
        weight_sum += w_map

    weight_sum = np.maximum(weight_sum, 1e-8)
    hdr /= weight_sum[..., None]

    return hdr

def read_exr_with_metadata(filepath):
    """Read EXR and return image and metadata."""
    print(f"Reading EXR: {filepath}")
    with OpenEXR.File(filepath) as infile:
        img = infile.channels()["RGB"].pixels  # shape: (H, W, 3)
        metadata = infile.header()
    return img, metadata

def read_exr_with_display_window(filepath):
    """
    Read EXR and return image, display window, and data window.
    Returns:
        img: np.ndarray, image data
        display_window: ((xmin, ymin), (xmax, ymax))
        data_window: ((oxmin, oymin), (oxmax, oymax))
    """
    with OpenEXR.File(filepath) as infile:
        img = infile.channels()["RGB"].pixels
        header = infile.header()
        display_window = header.get("displayWindow", None)
        # data_window = header.get("dataWindow", None)
    return img, display_window

def find_global_maps(input_dir):
    """Finds the undistort and vignette maps in the directory."""
    stmap_path = None
    vignette_path = None
    for f in os.listdir(input_dir):
        if f.endswith("_undistort_map.exr") and f[0] != ".":
            stmap_path = os.path.join(input_dir, f)
        elif f.endswith("_vignette_map.exr") and f[0] != ".":
            vignette_path = os.path.join(input_dir, f)
    return stmap_path, vignette_path

def pad_to_display_window(img, display_window):
    # display_window: ((xmin, ymin), (xmax, ymax))
    h, w = img.shape[:2]
    (xy_min, xy_max) = display_window
    xmin, ymin = xy_min
    xmax, ymax = xy_max
    out_h = ymax - ymin + 1
    out_w = xmax - xmin + 1
    padded = np.zeros((out_h, out_w, img.shape[2]), dtype=img.dtype)
    y_off = -ymin
    x_off = -xmin
    padded[y_off:y_off+h, x_off:x_off+w, :] = img
    return padded

def apply_vignette(image, vignette_map):
    return image * vignette_map

def apply_stmap(image, stmap):
    h, w = image.shape[:2]
    stmap_flipped = stmap.copy()
    stmap_flipped[..., 1] = 1.0 - stmap_flipped[..., 1]  # Flip Y for OpenCV
    # Add 0.5 for Nuke-style pixel center convention
    map_x = stmap_flipped[..., 0] * (w - 1) - 0.5
    map_y = stmap_flipped[..., 1] * (h - 1) - 0.5
    remapped = cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )
    return remapped

def merge_hdr_opencv(images, exposures, max_val=2.36, align=True):
    """
    Prepare linear float images for OpenCV HDR merge and merge using Debevec.
    images: numpy array (N, H, W, 3), float32, linear, range [0, max_val]
    exposures: list/array of exposure times in seconds
    max_val: maximum value for scaling (default 2.36)
    Returns: merged HDR image (float32, linear, RGB)
    """

    images_bgr_8u = []
    for img in images:
        # Scale to [0, 1]
        img_scaled = np.clip(img / max_val, 0, 1)
        # Gamma encode (sRGB)
        img_gamma = img_scaled ** (1/2.2)

        # Convert to 8-bit
        img_8u = (img_gamma * 255).round().astype(np.uint8)
        # Convert RGB to BGR
        img_bgr = img_8u[..., ::-1]
        images_bgr_8u.append(img_bgr)

    exposures = np.array(exposures, dtype=np.float32)

    # Align images if requested
    if align:

        align_mtb = cv2.createAlignMTB()

        try:
            align_mtb.process(images_bgr_8u, images_bgr_8u)
        except Exception as e:
            print(f"Error during alignment: {e}")




    # OpenCV expects a list of 8-bit BGR images
    merge_debevec = cv2.createMergeDebevec()
    hdr_bgr = merge_debevec.process(images_bgr_8u, times=exposures)

    # Convert back to RGB for saving
    hdr_rgb = hdr_bgr[..., ::-1]

    hdr_rgb = np.clip(hdr_rgb / 255.0, 0, 1) * max_val  # Scale to [0, max_val]

    hdr_rgb = hdr_rgb ** (2.2)

    return hdr_rgb

def save_exr_with_display_window(filepath, img, display_window):
    """Save numpy array as EXR using new API, with optional display window."""
    channels = {"RGB": img.astype(np.float32)}
    header = {
        "compression": OpenEXR.ZIP_COMPRESSION,
        "type": OpenEXR.scanlineimage,
    }
    h, w = img.shape[:2]
    if display_window is not None:
        header["displayWindow"] = display_window
        # Correct: dataWindow as ((xmin, ymin), (xmax, ymax))
        header["dataWindow"] = ( (0, 0), (w-1, h-1) )
    with OpenEXR.File(header, channels) as outfile:
        outfile.write(filepath)
    print(f"[✓] Saved merged EXR: {filepath}")

def save_exr(filepath, img):
    """Save numpy array as EXR using new API."""
    channels = {"RGB": img.astype(np.float32)}
    header = {
        "compression": OpenEXR.ZIP_COMPRESSION,
        "type": OpenEXR.scanlineimage,
    }
    with OpenEXR.File(header, channels) as outfile:
        outfile.write(filepath)
    print(f"[✓] Saved merged EXR: {filepath}")

def process_hdr(input_dir, mode="debevec"):
    exr_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith("_ACEScg.exr") and f[0] != "."
    ]
    if not exr_files:
        print("No EXR files found.")
        return

    # --- Detect and load correction maps ---
    stmap_path, vignette_path = find_global_maps(input_dir)
    stmap, display_window = None, None
    vignette = None
    if stmap_path:
        stmap, display_window = read_exr_with_display_window(stmap_path)
        stmap = stmap[..., :2]  # Use first two channels for STmap
        stmap_flipped = stmap.copy()
        stmap_flipped[..., 1] = 1.0 - stmap_flipped[..., 1]
    if vignette_path:
        vignette, _ = read_exr_with_display_window(vignette_path)

    # --- Read, correct, and pad images ---
    images = []
    exposures = []
    for f in exr_files:
        img, _ = read_exr_with_display_window(f)
        if vignette is not None:
            img = apply_vignette(img, vignette)
        if display_window is not None:
            img = pad_to_display_window(img, display_window)
        if stmap is not None:
            img = apply_stmap(img, stmap)
        images.append(img)
        _, meta = read_exr_with_metadata(f)
        exp = meta.get("exr/exif/2/exposure_time", 1.0)
        exposures.append(float(exp))

    images = np.stack(images, axis=0)  # shape: (N, H, W, 3)

    if mode == "debevec":
        merged = merge_hdr_linear(images, exposures, weighting='debevec')
    elif mode == "tent":
        merged = merge_hdr_linear(images, exposures, weighting='tent')
    elif mode == "uniform":
        merged = merge_hdr_linear(images, exposures, weighting='uniform')
    elif mode == "opencv":
        merged = merge_hdr_opencv(images, exposures)
    else:
        raise ValueError(f"Unsupported merge mode: {mode}")

    out_path = os.path.join(input_dir, "HDR_merged.exr")
    save_exr_with_display_window(out_path, merged, display_window)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge bracketed EXRs into HDR.")
    parser.add_argument("input_dir", help="Directory containing _ACEScg.exr files")
    parser.add_argument("--mode", choices=["debevec", "tent", "uniform", "opencv"], default="debevec", help="Merge mode")
    args = parser.parse_args()
    process_hdr(args.input_dir, mode=args.mode)