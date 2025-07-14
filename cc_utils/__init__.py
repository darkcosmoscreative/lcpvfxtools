import os
import sys
import numpy as np
import OpenEXR
import Imath
import rawpy


print('lcpvfxtools.cc_utils initialised')

# initialise config from module on path
current_dir = os.path.dirname(__file__)
module_root = os.path.abspath(os.path.join(current_dir, '../..'))
if module_root not in sys.path:
    sys.path.append(module_root)

from lcpvfxtools import config as cfg

# Matrix from ACES2065-1 (AP0) to ACEScg (AP1)
AP0_to_AP1 = np.array([
    [ 1.45143932, -0.23651075, -0.21492857],
    [-0.07655377,  1.1762297 , -0.09967593],
    [ 0.00831615, -0.00603245,  0.9977163 ]
])


def write_exr_from_cameraraw(write_dir, basename, raw_file_path, lens_dict=None):
    """
    Converts a camera RAW file to an OpenEXR file in XYZ color space.

    Args:
        write_dir (str): Directory to save the output EXR file.
        basename (str): Base name for the output file.
        raw_file_path (str): Path to the input RAW file.

    Returns:
        str: Path to the written EXR file.
    """
    # get extension and check raw file has valid extension using config global
    _, ext = os.path.splitext(raw_file_path)
    ext = ext[1:].lower()
    if ext in cfg.SUPPORTED_RAW_FORMATS:
    
       pass
    else:
        print(f"Unsupported RAW file format: {ext}. Supported formats are: {cfg.SUPPORTED_RAW_FORMATS}")
        return None

    # Read and process RAW file
    with rawpy.imread(raw_file_path) as raw:
        aces = raw.postprocess(
            output_color=rawpy.ColorSpace.ACES,
            gamma=(1, 1),                # Linear
            no_auto_bright=True,         # Preserve superbrights
            output_bps=16,               # RawPy internal bit depth; final is float16
            use_camera_wb=True           # Use in-camera white balance
        )

    aces = np.clip(aces, 0.0, None)

    # First scale (preserves midtones)
    scale = np.percentile(aces, 99.9)
    aces /= scale

    # Then apply highlight rolloff (only for values >1.0)
    def highlight_rolloff(x, threshold=1.0, softness=6.0):
        return np.where(
            x <= threshold,
            x,
            threshold + np.log1p((x - threshold) * softness) / softness
        )

    aces = highlight_rolloff(aces)
    aces *= 2.35


    # Assume aces is (H, W, 3)
    h, w = aces.shape[:2]

    # Reshape to (N, 3)
    aces_flat = aces.reshape(-1, 3)

    # Apply the matrix
    aces_flat = aces_flat @ AP0_to_AP1.T  # Transpose the matrix for correct multiplication

    # Reshape back to (H, W, 3)
    aces = aces_flat.reshape(h, w, 3)


    # Split into R, G, B channels and convert to half-float bytes
    r = aces[:, :, 0].astype(np.float16).tobytes()
    g = aces[:, :, 1].astype(np.float16).tobytes()
    b = aces[:, :, 2].astype(np.float16).tobytes()

    # Image dimensions
    height, width = aces.shape[:2]
    header = OpenEXR.Header(width, height)

    # Set channel types: 16-bit half-float
    
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    
    header['channels'] = {
        'R': half_chan,
        'G': half_chan,
        'B': half_chan
    }

    # Optional: embed basic metadata
    #header['owner'] = str(lens_dict['cam_maker'])

    # Create output path
    out_path = os.path.join(write_dir, f"{basename}_ACEScg.exr")

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
    x_pix = (xy[..., 0]) + Xc
    y_pix = (xy[..., 1]) + Yc
    s = (x_pix + 0.5) / w
    t = 1.0 - ((y_pix + 0.5) / h)
    return np.stack([s, t], axis=-1).astype(np.float32)


def save_st_exr(filepath, map_array, orig_width, orig_height, min_x, max_x, min_y, max_y):
    """
    Save a 2-channel map as an OpenEXR file with custom data and display windows.

    Args:
        filepath (str): Output file path.
        map_array (np.ndarray): Array of shape (H, W, 2).
        orig_width (int): Original image width (display window).
        orig_height (int): Original image height (display window).
        min_x (int): X offset (in pixels) of the overscan grid's top-left relative to original image.
        min_y (int): Y offset (in pixels) of the overscan grid's top-left relative to original image.
    """
    height, width = map_array.shape[:2]

    # Display window: original image bounds
    display_window_min = (0, 0)
    display_window_max = (orig_width - 1, orig_height - 1)

    # Data window: bounds of the overscan grid in original image coordinates
    data_window_min = min_x, min_y # (int(min_x), int(min_y))
    data_window_max = max_x, max_y #(int(min_x + width - 1), int(min_y + height - 1))

    '''
    print(f"Saving EXR: {filepath}")
    print(f"  map_array.shape: {map_array.shape}")
    print(f"  display_window_min: {display_window_min}")
    print(f"  display_window_max: {display_window_max}")
    print(f"  data_window_min: {data_window_min}")
    print(f"  data_window_max: {data_window_max}")
    '''

    header = OpenEXR.Header(orig_width, orig_height)
    header['displayWindow'] = Imath.Box2i(Imath.V2i(*display_window_min), Imath.V2i(*display_window_max))
    header['dataWindow'] = Imath.Box2i(Imath.V2i(*data_window_min), Imath.V2i(*data_window_max))

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    out = OpenEXR.OutputFile(filepath, header)
    R = map_array[:, :, 0].astype(np.float32).tobytes()
    G = map_array[:, :, 1].astype(np.float32).tobytes()
    out.writePixels({'R': R, 'G': G})
    out.close()
    print(f"Saved: {filepath}")

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


    xgrid_nopad, ygrid_nopad = np.meshgrid(
        np.arange(w),
        np.arange(h),
        indexing='xy'
    )

    x_nopad = xgrid_nopad - Xc
    y_nopad = ygrid_nopad - Yc
    grid_nopad = np.stack([x_nopad, y_nopad], axis=-1)

    k1 = radialdistortparam1 if radialdistortparam1 is not None else 0.0
    k2 = radialDistortparam2 if radialDistortparam2 is not None else 0.0
    k3 = radialDistortparam3 if radialDistortparam3 is not None else 0.0
    k4 = tangentialdistortparam1 if tangentialdistortparam1 is not None else 0.0
    k5 = tangentialdistortparam2 if tangentialdistortparam2 is not None else 0.0

    # do distortion for overscan first
    undist_nopad = get_reverse_geometry_distortion(
        grid_nopad, k1, k2, k3, k4, k5, focal_length_x, focal_length_y, Dmax)
    redist_nopad = get_geometry_distortion(
        grid_nopad, k1, k2, k3, k4, k5, focal_length_x, focal_length_y, Dmax)

    delta = redist_nopad - grid_nopad
    displacement = np.linalg.norm(delta, axis=-1)
    max_disp = ((np.max(displacement)) / max(w, h)) * 2
    #print(f"Max displacement: {max_disp:.4f}")
    overscan = ((1.0 + max_disp) / 1.0)
    #print(f"Calculated overscan factor: {overscan:.4f}")


    w_padded = int(np.ceil(w * overscan))
    if w_padded % 2 != 0:
        w_padded += 1  # Force even

    h_padded = int(np.ceil(h * overscan))
    if h_padded % 2 != 0:
        h_padded += 1  # Force even

    padded_w_total = abs(w_padded - w)
    padded_w_xmin = 0-int(padded_w_total // 2)
    padded_w_xmax = (w - 1) + int(padded_w_total // 2)
    padded_h_total = abs(h_padded - h)
    padded_h_ymin = 0-int(padded_h_total // 2)
    padded_h_ymax = (h - 1) + int(padded_h_total // 2)
    Xc_padded = w_padded / 2.0
    Yc_padded = h_padded / 2.0

    x_pix, y_pix = np.meshgrid(np.arange(w_padded), np.arange(h_padded), indexing='xy')
    x = x_pix - Xc_padded
    y = y_pix - Yc_padded
    grid = np.stack([x, y], axis=-1)

    undist_coords = get_reverse_geometry_distortion(
        grid, k1, k2, k3, k4, k5, focal_length_x, focal_length_y, Dmax)
    redist_coords = get_geometry_distortion(
        grid, k1, k2, k3, k4, k5, focal_length_x, focal_length_y, Dmax)

    undistort_map = to_nuke_stmap(undist_coords, w, h)
    distort_map = to_nuke_stmap(redist_coords, w, h)


    if write_dir and basename:
        # note - naming is reversed at the file system compared to calculation
        undistort_path = os.path.join(write_dir, f"{basename}_redistort_map.exr")
        distort_path = os.path.join(write_dir, f"{basename}_undistort_map.exr")
        save_st_exr(undistort_path, undistort_map, w, h, padded_w_xmin, padded_w_xmax, padded_h_ymin, padded_h_ymax)
        save_st_exr(distort_path, distort_map, w, h, padded_w_xmin, padded_w_xmax, padded_h_ymin, padded_h_ymax)

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

    '''
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
    '''

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

    # Convert to 3-channel map (using 1/channel to multiply in nuke)
    vignette_map = np.stack([1/gain, 1/gain, 1/gain], axis=-1).astype(np.float32)

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
    x_pix, y_pix = np.meshgrid(np.arange(w),
                               np.arange(h),
                               indexing='xy')
    x = x_pix - Xc
    y = y_pix - Yc

    grid = np.stack([x, y], axis=-1)



    # Red channel (vs green)
    k1_r = tca_redgreen_radial1 or 0.0
    k2_r = tca_redgreen_radial2 or 0.0
    k3_r = tca_redgreen_radial3 or 0.0


    red_distorted = get_geometry_distortion(
        grid, k1_r, k2_r, k3_r, 0.0, 0.0, focal_length_x, focal_length_y, Dmax)
    
    # Green channel (reference channel, red and blue are distorted relative to it)
    '''''
    # Green channel
    k1_g = tca_green_radial1 or 0.0
    k2_g = tca_green_radial2 or 0.0
    k3_g = tca_green_radial3 or 0.0

    green_distorted = get_geometry_distortion(
        grid, k1_g, k2_g, k3_g, 0.0, 0.0, focal_length_x, focal_length_y, Dmax)
    '''

    # Blue channel (vs green)
    k1_b = tca_bluegreen_radial1 or 0.0
    k2_b = tca_bluegreen_radial2 or 0.0
    k3_b = tca_bluegreen_radial3 or 0.0


    blue_distorted = get_geometry_distortion(
        grid, k1_b, k2_b, k3_b, 0.0, 0.0, focal_length_x, focal_length_y, Dmax)
    

    
    red_map = to_nuke_stmap(red_distorted, w, h)
    #green_map = to_nuke_stmap(green_distorted, w, h)
    blue_map = to_nuke_stmap(blue_distorted, w, h)



    if write_dir and basename:
        save_st_exr(os.path.join(write_dir, f"{basename}_tca_red.exr"), red_map, w, h, 0, w-1, 0, h-1)
        #save_st_exr(os.path.join(write_dir, f"{basename}_tca_green.exr"), green_map, w, h, 0, w-1, 0, h-1)
        save_st_exr(os.path.join(write_dir, f"{basename}_tca_blue.exr"), blue_map, w, h, 0, w-1, 0, h-1)
        print("Saved TCA maps for R, G, B.")

    return red_map, blue_map