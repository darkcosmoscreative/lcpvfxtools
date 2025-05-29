import os
import exifread
import sys

print("lcpvfxtools.exif_utils initialised")

# initialise config from module on path
current_dir = os.path.dirname(__file__)
module_root = os.path.abspath(os.path.join(current_dir, '../..'))
if module_root not in sys.path:
    sys.path.append(module_root)

from lcpvfxtools import config as cfg

def read_exif_data(file_path):
    """
    Read EXIF metadata from a file.

    Args:
        file_path (str): Path to the image or raw file.

    Returns:
        dict: Dictionary of EXIF tags and values. Empty if no EXIF or error.
    """
    try:
        with open(file_path, "rb") as f:
            tags = exifread.process_file(f)
        if tags is not None:
            return tags
        else:
            print("No EXIF data found.")
            return {}
    except Exception as e:
        print(f"Error reading EXIF data: {e}")
        return {}

def get_camera_lens_dict(rawfile=None):
    """
    Extract camera and lens information from a raw file's EXIF data.

    Args:
        rawfile (str): Path to the raw file.

    Returns:
        dict: Dictionary with keys 'cam_maker', 'cam_model', 'lens_maker', 'lens_model',
              'focal_length', 'aperture', and 'distance'.
    """
    lens_dict = {}
    cam_maker = None
    cam_model = None
    lens_maker = None
    lens_model = None
    focal_length = None
    aperture = None
    distance = None

    if rawfile and os.path.exists(rawfile):
        exif = read_exif_data(rawfile)
        for tag, value in exif.items():
            if('Image Make' in tag):
                cam_maker = str(value)
            elif('Image Model' in tag):
                cam_model = str(value)
            elif('LensMake' in tag):
                lens_maker = str(value)
            elif('LensModel' in tag):
                lens_model = str(value)
            elif('FocalLength' in tag):
                focal_string = f'{value}'
                if '/' in focal_string:
                    if not focal_length:
                        focal_length = eval(focal_string)
                else:
                    if not focal_length:
                        focal_length = float(focal_string)
            elif('EXIF ApertureValue' in tag):
                aperture_string = f'{value}'
                if '/' in aperture_string:
                    if not aperture:
                        aperture = eval(aperture_string)
                else:
                    if not aperture:
                        aperture = float(aperture_string)
            elif('Focus' in f'{tag}'):
                distance_string = f'{value}'
                if '/' in distance_string:
                    if not distance:
                        distance = eval(distance_string)
                elif 'Unknown' in distance_string:
                    distance = None
                else:
                    try:
                        if not distance:
                            distance = float(distance_string)
                    except:
                        distance = None
            elif('retty' in f'{tag}'):
                print(f'Pretty name: {value}')
            else:
                #print(f'Unknown tag: {tag} = {value}')
                pass
        if not distance:
            distance = 1.0

        lens_dict['cam_maker'] = cam_maker
        lens_dict['cam_model'] = cam_model
        lens_dict['lens_maker'] = lens_maker
        lens_dict['lens_model'] = lens_model
        lens_dict['focal_length'] = focal_length
        lens_dict['aperture'] = aperture
        lens_dict['distance'] = distance
    return lens_dict

def get_resolution_from_exif(rawfile=None):
    """
    Get the image resolution (width, height) from a raw file.

    Args:
        rawfile (str): Path to the raw file.

    Returns:
        tuple: (x_resolution, y_resolution) as integers, or (None, None) if not found.
    """
    x_resolution = None
    y_resolution = None
    try:
        import rawpy
        with rawpy.imread(rawfile) as raw:
            x_resolution, y_resolution = raw.raw_image_visible.shape[::-1]  # width, height
    except Exception:
        try:
            import cv2
            img = cv2.imread(rawfile, cv2.IMREAD_UNCHANGED)
            if img is not None:
                x_resolution, y_resolution = img.shape[1], img.shape[0]
        except Exception:
            pass
    return x_resolution, y_resolution