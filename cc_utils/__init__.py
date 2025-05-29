import os
import numpy as np
import OpenEXR
import Imath
import rawpy

print('lcpvfxtools.cc_utils initialised')


def write_exr_from_cameraraw(write_dir, basename, raw_file_path):
    """
    Converts a camera RAW file to an OpenEXR file in XYZ color space.

    Args:
        write_dir (str): Directory to save the output EXR file.
        basename (str): Base name for the output file.
        raw_file_path (str): Path to the input RAW file.

    Returns:
        str: Path to the written EXR file.
    """
    # Read and process RAW file
    with rawpy.imread(raw_file_path) as raw:
        xyz = raw.postprocess(
            output_color=rawpy.ColorSpace.XYZ,
            gamma=(1, 1),                # Linear
            no_auto_bright=True,         # Preserve superbrights
            output_bps=16,               # RawPy internal bit depth; final is float16
            use_camera_wb=True           # Use in-camera white balance
        )

    xyz = np.clip(xyz, 0.0, None)

    # First scale (preserves midtones)
    scale = np.percentile(xyz, 99.9)
    xyz /= scale

    # Then apply highlight rolloff (only for values >1.0)
    def highlight_rolloff(x, threshold=1.0, softness=6.0):
        return np.where(
            x <= threshold,
            x,
            threshold + np.log1p((x - threshold) * softness) / softness
        )

    xyz = highlight_rolloff(xyz)
    xyz *= 2.35

    # Split into R, G, B channels and convert to half-float bytes
    r = xyz[:, :, 0].astype(np.float16).tobytes()
    g = xyz[:, :, 1].astype(np.float16).tobytes()
    b = xyz[:, :, 2].astype(np.float16).tobytes()

    # Image dimensions
    height, width = xyz.shape[:2]
    header = OpenEXR.Header(width, height)

    # Set channel types: 16-bit half-float
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    header['channels'] = {
        'R': half_chan,
        'G': half_chan,
        'B': half_chan
    }

    # Optional: embed basic metadata
    header['comments'] = f'Converted from RAW: {os.path.basename(raw_file_path)}'

    # Create output path
    out_path = os.path.join(write_dir, f"{basename}_xyz.exr")

    # Write EXR
    exr = OpenEXR.OutputFile(out_path, header)
    exr.writePixels({'R': r, 'G': g, 'B': b})
    exr.close()

    print(f"[✓] Wrote: {out_path}")
    return out_path


def get_geometry_distortion(xy, k1, k2, k3, k4=0.0, k5=0.0, focal_length_x=1.0, focal_length_y=1.0, Dmax=1.0):
    """
    Apply rectilinear radial and tangential distortion to pixel-centered coordinates.

    Args:
        xy (np.ndarray): (..., 2) pixel-centered coordinates (x, y), centered at (0,0).
        k1, k2, k3 (float): Radial distortion coefficients.
        k4, k5 (float): Tangential distortion coefficients.
        focal_length_x, focal_length_y (float): Focal lengths from LCP.
        Dmax (float): Maximum of image width or height.

    Returns:
        np.ndarray: Distorted coordinates, same shape as xy.
    """
    x = xy[..., 0] / (focal_length_x * Dmax)
    y = xy[..., 1] / (focal_length_y * Dmax)
    r2 = x**2 + y**2
    radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
    x_dist = x * radial + 2*k4*x*y + k5*(r2 + 2*x**2)
    y_dist = y * radial + 2*k5*x*y + k4*(r2 + 2*y**2)
    x_dist = x_dist * (focal_length_x * Dmax)
    y_dist = y_dist * (focal_length_y * Dmax)
    return np.stack([x_dist, y_dist], axis=-1)

def get_reverse_geometry_distortion(xy, k1, k2, k3, k4=0.0, k5=0.0, focal_length_x=1.0, focal_length_y=1.0, Dmax=1.0, iterations=5):
    """
    Invert rectilinear radial and tangential distortion using an iterative method.

    Args:
        xy (np.ndarray): (..., 2) pixel-centered coordinates (x, y), centered at (0,0).
        k1, k2, k3 (float): Radial distortion coefficients.
        k4, k5 (float): Tangential distortion coefficients.
        focal_length_x, focal_length_y (float): Focal lengths from LCP.
        Dmax (float): Maximum of image width or height.
        iterations (int): Number of iterations for the inversion.

    Returns:
        np.ndarray: Undistorted coordinates, same shape as xy.
    """
    x_dist = xy[..., 0]
    y_dist = xy[..., 1]
    x_undist = x_dist.copy()
    y_undist = y_dist.copy()
    for _ in range(iterations):
        x = x_undist / (focal_length_x * Dmax)
        y = y_undist / (focal_length_y * Dmax)
        r2 = x**2 + y**2
        radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
        delta_x = 2*k4*x*y + k5*(r2 + 2*x**2)
        delta_y = 2*k5*x*y + k4*(r2 + 2*y**2)
        x_dist_pred = (x * radial + delta_x) * (focal_length_x * Dmax)
        y_dist_pred = (y * radial + delta_y) * (focal_length_y * Dmax)
        x_undist = x_undist - (x_dist_pred - x_dist)
        y_undist = y_undist - (y_dist_pred - y_dist)
    return np.stack([x_undist, y_undist], axis=-1)

def to_nuke_stmap(xy, w, h):
    """
    Convert pixel coordinates (centered at image center) to Nuke S/T map ([0,1], y-flipped).

    Args:
        xy (np.ndarray): (..., 2) pixel coordinates, centered at (0,0).
        w (int): Image width.
        h (int): Image height.

    Returns:
        np.ndarray: S/T map, shape (..., 2), dtype float32.
    """
    Xc = w / 2.0
    Yc = h / 2.0
    x_pix = xy[..., 0] + Xc
    y_pix = xy[..., 1] + Yc
    s = x_pix / w
    t = 1.0 - (y_pix / h)
    return np.stack([s, t], axis=-1).astype(np.float32)

def write_st_maps_from_params(write_dir=None,
                        basename=None,
                        x_resolution=None,
                        y_resolution=None,
                        focal_length_x=None,
                        focal_length_y=None,
                        radialdistortparam1=None,
                        radialDistortparam2=None,
                        radialDistortparam3=None,
                        tangentialdistortparam1=0.0,
                        tangentialdistortparam2=0.0):
    """
    Generate and save distortion and undistortion ST maps as EXR files.

    Args:
        write_dir (str): Directory to save the output files.
        basename (str): Base name for the output files.
        x_resolution (int): Image width.
        y_resolution (int): Image height.
        focal_length_x, focal_length_y (float): Focal lengths from LCP.
        radialdistortparam1, radialDistortparam2, radialDistortparam3 (float): Radial distortion coefficients.
        tangentialdistortparam1, tangentialdistortparam2 (float): Tangential distortion coefficients.

    Returns:
        tuple: (undistort_map, distort_map) as numpy arrays.
    """
    print('write_st_maps_from_params')
    print(f'write_dir: {write_dir}')
    print(f'basename: {basename}')
    print(f'x_resolution: {x_resolution}')
    print(f'y_resolution: {y_resolution}')
    print(f'focal_length_x: {focal_length_x}')
    print(f'focal_length_y: {focal_length_y}')
    print(f'radialdistortparam1: {radialdistortparam1}')
    print(f'radialDistortparam2: {radialDistortparam2}')
    print(f'radialDistortparam3: {radialDistortparam3}')
    print(f'tangentialdistortparam1: {tangentialdistortparam1}')
    print(f'tangentialdistortparam2: {tangentialdistortparam2}')

    h, w = int(y_resolution), int(x_resolution)
    Dmax = max(w, h)
    Xc = w / 2.0
    Yc = h / 2.0

    # Pixel grid, centered at image center
    x_pix, y_pix = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    x = x_pix - Xc
    y = y_pix - Yc
    grid = np.stack([x, y], axis=-1)

    k1 = radialdistortparam1 if radialdistortparam1 is not None else 0.0
    k2 = radialDistortparam2 if radialDistortparam2 is not None else 0.0
    k3 = radialDistortparam3 if radialDistortparam3 is not None else 0.0
    k4 = tangentialdistortparam1 if tangentialdistortparam1 is not None else 0.0
    k5 = tangentialdistortparam2 if tangentialdistortparam2 is not None else 0.0

    undist_coords = get_reverse_geometry_distortion(
        grid, k1, k2, k3, k4, k5, focal_length_x, focal_length_y, Dmax)
    redist_coords = get_geometry_distortion(
        grid, k1, k2, k3, k4, k5, focal_length_x, focal_length_y, Dmax)

    undistort_map = to_nuke_stmap(undist_coords, w, h)
    distort_map = to_nuke_stmap(redist_coords, w, h)

    def save_exr(filepath, map_array):
        """
        Save a 2-channel map as an OpenEXR file.

        Args:
            filepath (str): Output file path.
            map_array (np.ndarray): Array of shape (H, W, 2).
        """
        height, width = map_array.shape[:2]
        header = OpenEXR.Header(width, height)
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        out = OpenEXR.OutputFile(filepath, header)
        R = map_array[:, :, 0].astype(np.float32).tobytes()
        G = map_array[:, :, 1].astype(np.float32).tobytes()
        out.writePixels({'R': R, 'G': G})
        out.close()
        print(f"Saved: {filepath}")

    if write_dir and basename:
        undistort_path = os.path.join(write_dir, f"{basename}_undistort_map.exr")
        distort_path = os.path.join(write_dir, f"{basename}_distort_map.exr")
        save_exr(undistort_path, undistort_map)
        save_exr(distort_path, distort_map)

        # Write a txt file with the parameters used
        params_path = os.path.join(write_dir, f"{basename}_st_map_params.txt")
        with open(params_path, "w") as f:
            f.write("ST Map Generation Parameters\n")
            f.write(f"Resolution: {w} x {h}\n")
            f.write(f"FocalLengthX: {focal_length_x}\n")
            f.write(f"FocalLengthY: {focal_length_y}\n")
            f.write(f"RadialDistortParam1: {radialdistortparam1}\n")
            f.write(f"RadialDistortParam2: {radialDistortparam2}\n")
            f.write(f"RadialDistortParam3: {radialDistortparam3}\n")
            f.write(f"TangentialDistortParam1: {tangentialdistortparam1}\n")
            f.write(f"TangentialDistortParam2: {tangentialdistortparam2}\n")
        print(f"Saved parameters to {params_path}")

    return undistort_map, distort_map

def write_vignette_map_from_params(write_dir=None,
                        basename=None,
                        x_resolution=None,
                        y_resolution=None,
                        focal_length_x=None,
                        focal_length_y=None,
                        vignette_param1=None,
                        vignette_param2=None,
                        vignette_param3=None):
    """
    Generate and save a vignette gain map as an EXR file.

    Args:
        write_dir (str): Directory to save the output file.
        basename (str): Base name for the output file.
        x_resolution (int): Image width.
        y_resolution (int): Image height.
        focal_length_x, focal_length_y (float): Focal lengths from LCP.
        vignette_param1, vignette_param2, vignette_param3 (float): Vignette polynomial coefficients.

    Returns:
        np.ndarray: Vignette gain map as a numpy array.
    """
    print('write_vignette_map_from_params')
    print(f'write_dir: {write_dir}')
    print(f'basename: {basename}')
    print(f'x_resolution: {x_resolution}')
    print(f'y_resolution: {y_resolution}')
    print(f'focal_length_x: {focal_length_x}')
    print(f'focal_length_y: {focal_length_y}')
    print(f'vignette_param1: {vignette_param1}')
    print(f'vignette_param2: {vignette_param2}')
    print(f'vignette_param3: {vignette_param3}')

    h, w = int(y_resolution), int(x_resolution)
    Dmax = max(w, h)
    Xc = w / 2.0
    Yc = h / 2.0

    # Create pixel grid, y flipped for Nuke-style images
    x_pix, y_pix = np.meshgrid(
        np.arange(w),
        np.arange(h)[::-1],
        indexing='xy'
    )
    x = x_pix - Xc
    y = y_pix - Yc

    # Normalize x and y as per Adobe's convention
    x_norm = x / (focal_length_x * Dmax)
    y_norm = y / (focal_length_y * Dmax)
    r = np.sqrt(x_norm**2 + y_norm**2)

    # Clip radius to [0, 1] for robustness (optional)
    r = np.clip(r, 0, 1)

    # Polynomial gain function from LCP model
    p1 = vignette_param1 if vignette_param1 is not None else 0.0
    p2 = vignette_param2 if vignette_param2 is not None else 0.0
    p3 = vignette_param3 if vignette_param3 is not None else 0.0

    gain = 1.0 + p1 * (r**2) + p2 * (r**4) + p3 * (r**6)

    # Clamp gain to avoid negative/unstable results (optional but wise)
    gain = np.clip(gain, 0.001, 10.0)

    # Convert to 3-channel map for RGB gain (optional)
    vignette_map = np.stack([gain, gain, gain], axis=-1).astype(np.float32)

    def save_exr(filepath, map_array):
        """
        Save a 3-channel map as an OpenEXR file.

        Args:
            filepath (str): Output file path.
            map_array (np.ndarray): Array of shape (H, W, 3).
        """
        height, width = map_array.shape[:2]
        header = OpenEXR.Header(width, height)
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        out = OpenEXR.OutputFile(filepath, header)
        R = map_array[:, :, 0].astype(np.float32).tobytes()
        G = map_array[:, :, 1].astype(np.float32).tobytes()
        B = map_array[:, :, 2].astype(np.float32).tobytes()
        out.writePixels({'R': R, 'G': G, 'B': B})
        out.close()

    if write_dir and basename:
        vignette_path = os.path.join(write_dir, f"{basename}_vignette_map.exr")
        save_exr(vignette_path, vignette_map)
        print(f"Saved vignette map to {vignette_path}")

    return vignette_map

def write_tca_maps_from_params(write_dir=None,
                        basename=None,
                        x_resolution=None,
                        y_resolution=None,
                        focal_length_x=None,
                        focal_length_y=None,
                        tca_redgreen_radial1=None,
                        tca_redgreen_radial2=None,
                        tca_redgreen_radial3=None,
                        tca_green_radial1=None,
                        tca_green_radial2=None,
                        tca_green_radial3=None,
                        tca_bluegreen_radial1=None,
                        tca_bluegreen_radial2=None,
                        tca_bluegreen_radial3=None):
    """
    Generate and save TCA (transverse chromatic aberration) maps for red and blue channels as EXR files.

    Args:
        write_dir (str): Directory to save the output files.
        basename (str): Base name for the output files.
        x_resolution (int): Image width.
        y_resolution (int): Image height.
        focal_length_x, focal_length_y (float): Focal lengths from LCP.
        tca_redgreen_radial1, tca_redgreen_radial2, tca_redgreen_radial3 (float): Red-green TCA coefficients.
        tca_green_radial1, tca_green_radial2, tca_green_radial3 (float): Green TCA coefficients (unused here).
        tca_bluegreen_radial1, tca_bluegreen_radial2, tca_bluegreen_radial3 (float): Blue-green TCA coefficients.

    Returns:
        tuple: (red_map, blue_map) as numpy arrays.
    """
    print('write_tca_maps_from_params')

    h, w = int(y_resolution), int(x_resolution)
    Dmax = max(w, h)
    Xc = w / 2.0
    Yc = h / 2.0

    # Pixel grid, centered at image center
    x_pix, y_pix = np.meshgrid(np.arange(w), np.arange(h)[::-1], indexing='xy')
    x = x_pix - Xc
    y = y_pix - Yc

    # Normalize as per Adobe's model (same as geometry distortion)
    x_norm = x / (focal_length_x * Dmax)
    y_norm = y / (focal_length_y * Dmax)
    r2 = x_norm**2 + y_norm**2

    def distortion_params(k1, k2, k3):
        return 1 + k1*r2 + k2*r2**2 + k3*r2**3

    # Red channel (vs green)
    k1_r = tca_redgreen_radial1 or 0.0
    k2_r = tca_redgreen_radial2 or 0.0
    k3_r = tca_redgreen_radial3 or 0.0
    scale_red = distortion_params(k1_r, k2_r, k3_r)
    map_x_red = x_norm * scale_red
    map_y_red = y_norm * scale_red

    # Blue channel (vs green)
    k1_b = tca_bluegreen_radial1 or 0.0
    k2_b = tca_bluegreen_radial2 or 0.0
    k3_b = tca_bluegreen_radial3 or 0.0
    scale_blue = distortion_params(k1_b, k2_b, k3_b)
    map_x_blue = x_norm * scale_blue
    map_y_blue = y_norm * scale_blue

    def to_st(x_map, y_map):
        """
        Convert normalized TCA coordinates back to pixel coordinates and then to S/T map.

        Args:
            x_map (np.ndarray): Normalized x coordinates.
            y_map (np.ndarray): Normalized y coordinates.

        Returns:
            np.ndarray: S/T map, shape (..., 2), dtype float32.
        """
        x_pix = x_map * (focal_length_x * Dmax) + Xc
        y_pix = y_map * (focal_length_y * Dmax) + Yc
        s = x_pix / w
        t = (y_pix / h)
        return np.stack([s, t], axis=-1).astype(np.float32)

    red_map = to_st(map_x_red, map_y_red)
    blue_map = to_st(map_x_blue, map_y_blue)

    def save_exr(filepath, map_array):
        """
        Save a 2-channel map as an OpenEXR file.

        Args:
            filepath (str): Output file path.
            map_array (np.ndarray): Array of shape (H, W, 2).
        """
        height, width = map_array.shape[:2]
        header = OpenEXR.Header(width, height)
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        out = OpenEXR.OutputFile(filepath, header)
        R = map_array[:, :, 0].tobytes()
        G = map_array[:, :, 1].tobytes()
        out.writePixels({'R': R, 'G': G})
        out.close()

    if write_dir and basename:
        save_exr(os.path.join(write_dir, f"{basename}_tca_red.exr"), red_map)
        save_exr(os.path.join(write_dir, f"{basename}_tca_blue.exr"), blue_map)
        print("Saved TCA maps for R, B.")

    return red_map, blue_map
