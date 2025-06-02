import os
import numpy as np
import OpenEXR
import Imath
import rawpy
import scipy.ndimage

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


def get_geometry_distortion(
    xy, k1, k2, k3, k4=0.0, k5=0.0,
    focal_length_x=1.0, focal_length_y=1.0, Dmax=1.0,
    overscan=1.0
):
    """
    Apply rectilinear radial and tangential distortion to pixel-centered coordinates.
    If overscan != 1.0, scale focal lengths and Dmax accordingly.

    Args:
        xy (np.ndarray): (..., 2) pixel-centered coordinates (x, y), centered at (0,0).
        k1, k2, k3 (float): Radial distortion coefficients.
        k4, k5 (float): Tangential distortion coefficients.
        focal_length_x, focal_length_y (float): Focal lengths from LCP.
        Dmax (float): Maximum of image width or height.
        overscan (float): Overscan scaling factor (default 1.0).

    Returns:
        np.ndarray: Distorted coordinates, same shape as xy.
    """

    xy_scaled = xy / overscan
    dmax_scaled = Dmax / overscan

    x = xy_scaled[..., 0] / (focal_length_x * Dmax)
    y = xy_scaled[..., 1] / (focal_length_y * Dmax)
    r2 = x**2 + y**2
    radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
    x_dist = x * radial + 2*k4*x*y + k5*(r2 + 2*x**2)
    y_dist = y * radial + 2*k5*x*y + k4*(r2 + 2*y**2)
    x_dist *= (focal_length_x * Dmax)
    y_dist *= (focal_length_y * Dmax)

    distorted_xy = np.stack([x_dist, y_dist], axis=-1)
    distorted_xy *= overscan
    return distorted_xy

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

def to_nuke_stmap(xy, w, h, x_min=None, x_max=None, y_min=None, y_max=None):
    """
    Convert pixel coordinates (centered at image center) to Nuke S/T map ([0,1], y-flipped).
    Supports grids that extend outside the original image bounds.

    Args:
        xy (np.ndarray): (..., 2) pixel coordinates, centered at (0,0).
        w (int): Output image width.
        h (int): Output image height.
        x_min, x_max, y_min, y_max (float, optional): Bounds of the grid in pixel space.
            If None, defaults to original image bounds.

    Returns:
        np.ndarray: S/T map, shape (..., 2), dtype float32.
    """
    print("to_nuke_stmap debug:")
    print(f"  xy.shape: {xy.shape}")
    print(f"  w, h: {w}, {h}")
    print(f"  x_min: {x_min}, x_max: {x_max}")
    print(f"  y_min: {y_min}, y_max: {y_max}")
    print(f"  xy[...,0] min/max: {xy[...,0].min()}, {xy[...,0].max()}")
    print(f"  xy[...,1] min/max: {xy[...,1].min()}, {xy[...,1].max()}")

    if x_min is None or x_max is None:
        Xc = w / 2.0
        x_pix = xy[..., 0] + Xc
        s = x_pix / w
        print(f"  Using default x bounds: Xc={Xc}")
    else:
        s = (xy[..., 0] - x_min) / (x_max - x_min)
        print(f"  Using custom x bounds: s.min={s.min()}, s.max={s.max()}")

    if y_min is None or y_max is None:
        Yc = h / 2.0
        y_pix = xy[..., 1] + Yc
        t = 1.0 - (y_pix / h)
        print(f"  Using default y bounds: Yc={Yc}")
    else:
        t = 1.0 - (xy[..., 1] - y_min) / (y_max - y_min)
        print(f"  Using custom y bounds: t.min={t.min()}, t.max={t.max()}")

    stmap = np.stack([s, t], axis=-1).astype(np.float32)
    print(f"  stmap.shape: {stmap.shape}, stmap.min: {stmap.min()}, stmap.max: {stmap.max()}")
    return stmap

def simulate_distortion_with_overscan(
    base_size=1000, overscan_percent=0.10, channels=2
):
    """
    Simulates a distorted image with a given percentage of overscan.
    
    Args:
        base_size (int): Width and height of the original square image.
        overscan_percent (float): Percent of overscan in both dimensions (e.g., 0.10 for 10%).
        channels (int): Number of image channels (e.g., 1 for grayscale, 2 for ST map).
    
    Returns:
        tuple: (original_array, distorted_array)
    """
    # Create original array (zeros)
    shape = (base_size, base_size, channels) if channels > 1 else (base_size, base_size)
    original = np.zeros(shape, dtype=np.float32)

    # Calculate new size with overscan
    new_size = int(base_size * (1 + overscan_percent))
    zoom_factor = new_size / base_size

    if channels > 1:
        zoom_factors = (zoom_factor, zoom_factor, 1)
    else:
        zoom_factors = zoom_factor

    # Use scipy.ndimage.zoom to scale up (simulate distortion overscan)
    distorted = scipy.ndimage.zoom(original, zoom=zoom_factors, order=1)

    return original, distorted


def compute_overscan_bounds(image_shape, distorted_coords):
    """
    Compute overscan bounds from a distorted coordinate map.

    Args:
        image_shape (tuple): (height, width) of the original image.
        distorted_coords (np.ndarray): Array of shape (H, W, 2) of distorted pixel positions.

    Returns:
        dict: Dictionary with all coordinate bounds in various spaces.
    """

    test_result = {
    'width': 1000,
    'height': 1000,
    'overscan_pixels_x': 10,
    'overscan_pixels_y': 10,
    'data_width': 1020,
    'data_height': 1020,

    # Display pixel bounds
    'min_display_pixel_x': 0,
    'max_display_pixel_x': 999,
    'min_display_pixel_y': 0,
    'max_display_pixel_y': 999,

    # Data pixel bounds (includes overscan)
    'min_data_pixel_x': -10,
    'max_data_pixel_x': 1009,
    'min_data_pixel_y': -10,
    'max_data_pixel_y': 1009,

    # Display-centered pixel bounds
    'min_display_centred_pixel_x': -500,
    'max_display_centred_pixel_x': 499,
    'min_display_centred_pixel_y': -500,
    'max_display_centred_pixel_y': 499,

    # Data-centered pixel bounds
    'min_data_centred_pixel_x': -510,
    'max_data_centred_pixel_x': 509,
    'min_data_centred_pixel_y': -510,
    'max_data_centred_pixel_y': 509,

    # Normalized float bounds (0–1 range)
    'min_display_float_x': 0.0,
    'max_display_float_x': 0.999,
    'min_display_float_y': 0.0,
    'max_display_float_y': 0.999,

    'min_data_float_x': -0.01,
    'max_data_float_x': 1.009,
    'min_data_float_y': -0.01,
    'max_data_float_y': 1.009,

    # Centered float coordinates (remapped to -1 to 1)
    'min_display_centred_float_x': -1.0,
    'max_display_centred_float_x': 1.0,
    'min_display_centred_float_y': -1.0,
    'max_display_centred_float_y': 1.0,

    'min_data_centred_float_x': -1.02,
    'max_data_centred_float_x': 1.018,
    'min_data_centred_float_y': -1.02,
    'max_data_centred_float_y': 1.018,

    # Center pixel location
    'centre_pixel_x': 500.0,
    'centre_pixel_y': 500.0,
}


    h, w = image_shape
    cx = w / 2.0
    cy = h / 2.0
    dist_w, dist_h = distorted_coords.shape[:2]
    cdist_x = dist_w / 2.0
    cdist_y = dist_h / 2.0

    #x_coords = distorted_coords[..., 0].flatten()
    #y_coords = distorted_coords[..., 1].flatten()

    min_data_pixel_x = 0
    max_data_pixel_x, max_data_pixel_y = distorted_coords.shape[:2]
    min_data_pixel_y = 0
    #max_data_pixel_y = float(np.max(y_coords))

    return {
        # Pixel space (uncentered)
        'min_data_pixel_x': min_data_pixel_x,
        'max_data_pixel_x': max_data_pixel_x,
        'min_data_pixel_y': min_data_pixel_y,
        'max_data_pixel_y': max_data_pixel_y,
        'min_display_pixel_x': 0.0,
        'max_display_pixel_x': w,
        'min_display_pixel_y': 0.0,
        'max_display_pixel_y': h,

        # Pixel space (centered)
        'min_data_centred_pixel_x': min_data_pixel_x - cdist_x,
        'max_data_centred_pixel_x': max_data_pixel_x - cdist_x,
        'min_data_centred_pixel_y': min_data_pixel_y - cdist_y,
        'max_data_centred_pixel_y': max_data_pixel_y - cdist_y,
        'min_display_centred_pixel_x': -cx,
        'max_display_centred_pixel_x': cx - 1,
        'min_display_centred_pixel_y': -cy,
        'max_display_centred_pixel_y': cy - 1,

        # Float normalized (0 to 1)
        'min_data_float_x': (1- (min_data_pixel_x - cdist_x) / -cx) / 2,  #min_data_pixel_x / w,
        'max_data_float_x': 1 + ((((min_data_pixel_x - cdist_x) / -cx) - 1)/2), #max_data_pixel_x / w,
        'min_data_float_y': (1- (min_data_pixel_y - cdist_y) / -cy) / 2, #min_data_pixel_y / h,
        'max_data_float_y': 1 + ((((min_data_pixel_y - cdist_y) / -cy) - 1)/2), #max_data_pixel_y / h,
        'min_display_float_x': 0.0,
        'max_display_float_x': 1.0,
        'min_display_float_y': 0.0,
        'max_display_float_y': 1.0,

        # Float normalized (centered: -1 to 1)
        'min_data_centred_float_x': (min_data_pixel_x - cdist_x) / cx,
        'max_data_centred_float_x': (max_data_pixel_x - cdist_x) / cx,
        'min_data_centred_float_y': (min_data_pixel_y - cdist_y) / cy,
        'max_data_centred_float_y': (max_data_pixel_y - cdist_y) / cy,
        'min_display_centred_float_x': -1.0,
        'max_display_centred_float_x': 1.0,
        'min_display_centred_float_y': -1.0,
        'max_display_centred_float_y': 1.0,

        # Center of image in pixels
        'centre_pixel_x': cx,
        'centre_pixel_y': cy,
    }




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
                        tangentialdistortparam2=0.0,
                        overscan=1.5):
    """
    Generate and save distortion and undistortion ST maps as EXR files, with overscan-aware grid.

    Args:
        write_dir (str): Directory to save the output files.
        basename (str): Base name for the output files.
        x_resolution (int): Image width.
        y_resolution (int): Image height.
        focal_length_x, focal_length_y (float): Focal lengths from LCP.
        radialdistortparam1, radialDistortparam2, radialDistortparam3 (float): Radial distortion coefficients.
        tangentialdistortparam1, tangentialdistortparam2 (float): Tangential distortion coefficients.
        overscan (float): Overscan factor for the initial grid (default 1.5).

    Returns:
        tuple: (undistort_map, distort_map) as numpy arrays.
    """
    print('write_st_maps_from_params (with overscan)')
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
    print(f'overscan: {overscan}')

    # --- 0. do some testing on bounds ---
    '''
    orig_test, dist_test = simulate_distortion_with_overscan()
    print(f"  [DEBUG] Original shape: {orig_test.shape[:2]}, Distorted shape: {dist_test.shape}")
    overscan_dict = compute_overscan_bounds(orig_test.shape[:2], distorted_coords=dist_test)
    for key, value in overscan_dict.items():
        print(f"  [DEBUG] {key}: {value}")
    '''

    h, w = int(y_resolution), int(x_resolution)
    Dmax = max(w, h)
    Xc = w / 2.0
    Yc = h / 2.0

    # --- 1. Create overscan grid ---
    w_os = int(w * overscan)
    h_os = int(h * overscan)
    Xc_os = w_os / 2.0
    Yc_os = h_os / 2.0

    x_pix_os, y_pix_os = np.meshgrid(np.arange(w_os), np.arange(h_os), indexing='xy')
    x_os = x_pix_os - Xc_os
    y_os = y_pix_os - Yc_os
    grid_os = np.stack([x_os, y_os], axis=-1)

    k1 = radialdistortparam1 if radialdistortparam1 is not None else 0.0
    k2 = radialDistortparam2 if radialDistortparam2 is not None else 0.0
    k3 = radialDistortparam3 if radialDistortparam3 is not None else 0.0
    k4 = tangentialdistortparam1 if tangentialdistortparam1 is not None else 0.0
    k5 = tangentialdistortparam2 if tangentialdistortparam2 is not None else 0.0

    # --- 2. Distort overscan grid ---
    redist_coords_os = get_geometry_distortion(
        grid_os, k1, k2, k3, k4, k5, focal_length_x, focal_length_y, Dmax, overscan
    )

    '''
    # --- 3. Find min/max bounds of distorted overscan grid ---
    print(f"  [DEBUG] redist_coords_os.shape: {redist_coords_os.shape}")
    print(f"  [DEBUG] w (image width): {w}, h (image height): {h}")
    print(f"  [DEBUG] w_os (overscan width): {w_os}, h_os (overscan height): {h_os}")
    print(f"  [DEBUG] grid_os[...,0] (x) min/max: {grid_os[...,0].min()}, {grid_os[...,0].max()}")
    print(f"  [DEBUG] grid_os[...,1] (y) min/max: {grid_os[...,1].min()}, {grid_os[...,1].max()}")
    print(f"  [DEBUG] redist_coords_os[...,0] (x) min/max: {redist_coords_os[...,0].min()}, {redist_coords_os[...,0].max()}")
    print(f"  [DEBUG] redist_coords_os[...,1] (y) min/max: {redist_coords_os[...,1].min()}, {redist_coords_os[...,1].max()}")

    min_x = np.min(redist_coords_os[..., 0])
    max_x = np.max(redist_coords_os[..., 0])
    min_y = np.min(redist_coords_os[..., 1])
    max_y = np.max(redist_coords_os[..., 1])
    print(f"  [DEBUG] min_x: {min_x}, max_x: {max_x}")
    print(f"  [DEBUG] min_y: {min_y}, max_y: {max_y}")
    '''

    redist_coords_os_dict = compute_overscan_bounds((h, w), distorted_coords=redist_coords_os)
    min_x = redist_coords_os_dict['min_data_centred_pixel_x']
    max_x = redist_coords_os_dict['max_data_centred_pixel_x']
    min_y = redist_coords_os_dict['min_data_centred_pixel_y']
    max_y = redist_coords_os_dict['max_data_centred_pixel_y']
    samples_x = redist_coords_os_dict['max_data_pixel_x']
    samples_y = redist_coords_os_dict['max_data_pixel_y']

    # --- 4. Create optimized grid covering only the needed bounds ---
    x_pix_opt = np.linspace(min_x, max_x, samples_x)
    y_pix_opt = np.linspace(min_y, max_y, samples_y)
    x_opt, y_opt = np.meshgrid(x_pix_opt, y_pix_opt, indexing='xy')
    grid_opt = np.stack([x_opt, y_opt], axis=-1)

    # --- 5. Distort and undistort the optimized grid ---
    undist_coords = get_reverse_geometry_distortion(
        grid_opt, k1, k2, k3, k4, k5, focal_length_x, focal_length_y, Dmax
    )
    redist_coords = get_geometry_distortion(
        grid_opt, k1, k2, k3, k4, k5, focal_length_x, focal_length_y, Dmax
    )
    '''
    # --- 6. calculate overscan bounds for the optimized grid ---
    overscan_bounds = compute_overscan_bounds((h, w), distorted_coords=redist_coords)
    for key, value in overscan_bounds.items():
        print(f"  [DEBUG] {key}: {value}")
    '''

    for key, value in redist_coords_os_dict.items():
        print(f"  [DEBUG] {key}: {value}")

    undistort_map = to_nuke_stmap(undist_coords, w, h, x_min=min_x, x_max=max_x, y_min=min_y, y_max=max_y)
    distort_map = to_nuke_stmap(redist_coords, w, h, x_min=min_x, x_max=max_x, y_min=min_y, y_max=max_y)

    def save_exr(filepath, map_array, display_width, display_height, min_x=0, min_y=0):
        """
        Save a 2-channel map as an OpenEXR file with custom data window (bounding box).
        Args:
            filepath (str): Output file path.
            map_array (np.ndarray): Array of shape (H, W, 2).
            display_width (int): Display window width (original image width).
            display_height (int): Display window height (original image height).
            min_x (int): X offset of the data window (bounding box).
            min_y (int): Y offset of the data window (bounding box).
        """
        height, width = map_array.shape[:2]

        # Calculate display window (original image)
        display_window_min = (0, 0)
        display_window_max = (display_width - 1, display_height - 1)

        # Calculate data window (bounding box for this map)
        data_window_min = (int(np.floor(min_x)), int(np.floor(min_y)))
        data_window_max = (int(np.floor(min_x + width - 1)), int(np.floor(min_y + height - 1)))

        # If data window is smaller than display window, clamp to display window
        dw_min_x = min(data_window_min[0], display_window_min[0])
        dw_min_y = min(data_window_min[1], display_window_min[1])
        dw_max_x = max(data_window_max[0], display_window_max[0])
        dw_max_y = max(data_window_max[1], display_window_max[1])

        # Debug prints
        print(f"Saving EXR: {filepath}")
        print(f"  map_array.shape: {map_array.shape}")
        print(f"  display_width, display_height: {display_width}, {display_height}")
        print(f"  display_window_min: {display_window_min}")
        print(f"  display_window_max: {display_window_max}")
        print(f"  data_window_min: {data_window_min}")
        print(f"  data_window_max: {data_window_max}")
        print(f"  final_data_window_min: ({dw_min_x}, {dw_min_y})")
        print(f"  final_data_window_max: ({dw_max_x}, {dw_max_y})")
        print(f"  Bytes per channel: {width * height * 4}")

        # Set display window and data window in header
        header = OpenEXR.Header(display_width, display_height)
        header['displayWindow'] = Imath.Box2i(Imath.V2i(*display_window_min), Imath.V2i(*display_window_max))
        header['dataWindow'] = Imath.Box2i(Imath.V2i(dw_min_x, dw_min_y), Imath.V2i(dw_max_x, dw_max_y))

        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        out = OpenEXR.OutputFile(filepath, header)
        R = map_array[:, :, 0].astype(np.float32).tobytes()
        G = map_array[:, :, 1].astype(np.float32).tobytes()
        print(f"  Actual bytes for R: {len(R)}")
        print(f"  Actual bytes for G: {len(G)}")
        out.writePixels({'R': R, 'G': G})
        out.close()
        print(f"Saved: {filepath}")

    if write_dir and basename:
        undistort_path = os.path.join(write_dir, f"{basename}_undistort_map.exr")
        distort_path = os.path.join(write_dir, f"{basename}_distort_map.exr")
        save_exr(undistort_path, undistort_map, w, h, min_x, min_y)
        save_exr(distort_path, distort_map, w, h, min_x, min_y)

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
            f.write(f"Overscan: {overscan}\n")
            f.write(f"Optimized grid bounds: x({min_x:.2f}, {max_x:.2f}), y({min_y:.2f}, {max_y:.2f})\n")
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

    def save_exr(filepath, map_array, display_width, display_height, min_x=0, min_y=0):
        """
        Save a 2-channel map as an OpenEXR file with custom data window (bounding box).
        Args:
            filepath (str): Output file path.
            map_array (np.ndarray): Array of shape (H, W, 2).
            display_width (int): Display window width (original image width).
            display_height (int): Display window height (original image height).
            min_x (int): X offset of the data window (bounding box).
            min_y (int): Y offset of the data window (bounding box).
        """
        height, width = map_array.shape[:2]

        # Calculate display window (original image)
        display_window_min = (0, 0)
        display_window_max = (display_width - 1, display_height - 1)

        # Calculate data window (bounding box for this map)
        data_window_min = (int(np.floor(min_x)), int(np.floor(min_y)))
        data_window_max = (int(np.floor(min_x + width - 1)), int(np.floor(min_y + height - 1)))

        # If data window is smaller than display window, clamp to display window
        dw_min_x = min(data_window_min[0], display_window_min[0])
        dw_min_y = min(data_window_min[1], display_window_min[1])
        dw_max_x = max(data_window_max[0], display_window_max[0])
        dw_max_y = max(data_window_max[1], display_window_max[1])

        # Debug prints
        print(f"Saving EXR: {filepath}")
        print(f"  map_array.shape: {map_array.shape}")
        print(f"  display_width, display_height: {display_width}, {display_height}")
        print(f"  display_window_min: {display_window_min}")
        print(f"  display_window_max: {display_window_max}")
        print(f"  data_window_min: {data_window_min}")
        print(f"  data_window_max: {data_window_max}")
        print(f"  final_data_window_min: ({dw_min_x}, {dw_min_y})")
        print(f"  final_data_window_max: ({dw_max_x}, {dw_max_y})")
        print(f"  Bytes per channel: {width * height * 4}")

        # Set display window and data window in header
        header = OpenEXR.Header(display_width, display_height)
        header['displayWindow'] = Imath.Box2i(Imath.V2i(*display_window_min), Imath.V2i(*display_window_max))
        header['dataWindow'] = Imath.Box2i(Imath.V2i(dw_min_x, dw_min_y), Imath.V2i(dw_max_x, dw_max_y))

        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        out = OpenEXR.OutputFile(filepath, header)
        R = map_array[:, :, 0].astype(np.float32).tobytes()
        G = map_array[:, :, 1].astype(np.float32).tobytes()
        print(f"  Actual bytes for R: {len(R)}")
        print(f"  Actual bytes for G: {len(G)}")
        out.writePixels({'R': R, 'G': G})
        out.close()
        print(f"Saved: {filepath}")

    if write_dir and basename:
        save_exr(os.path.join(write_dir, f"{basename}_tca_red.exr"), red_map, w, h, min_x, min_y)
        save_exr(os.path.join(write_dir, f"{basename}_tca_blue.exr"), blue_map, w, h, min_x, min_y)
        print("Saved TCA maps for R, B.")

    return red_map, blue_map
