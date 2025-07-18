import os
import exifread
import sys
import rawpy
import cv2

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
    exposure_time = None
    fnumber = None
    exposure_program = None
    iso_speed = None
    date_time_original = None
    shutter_speed_value = None
    exposure_bias_value = None
    metering_mode = None
    flash = None
    exposure_mode = None
    white_balance = None
    bracket_mode = None
    bracket_value = None
    bracket_shot_number = None

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
                #print(f'Pretty name: {value}')
                pass
            elif('ExposureTime' in f'{tag}'):
                #print(f'Exposure time: {value}')
                exposure_time = str(value)
            elif('FNumber' in f'{tag}'):
                #print(f'FNumber: {value}')
                fnumber = str(value)
            elif('ExposureProgram' in f'{tag}'):
                #print(f'Exposure program: {value}')
                exposure_program = str(value)
            elif('ISOSpeedRatings' in f'{tag}'):
                #print(f'ISO speed ratings: {value}')
                iso_speed = str(value)
            elif('DateTimeOriginal' in f'{tag}'):
                #print(f'DateTimeOriginal: {value}')
                date_time_original = str(value)
            elif('ShutterSpeedValue' in f'{tag}'):
                #print(f'ShutterSpeedValue: {value}')
                shutter_speed_value = str(value)
            elif('ExposureBiasValue' in f'{tag}'):
                #print(f'ExposureBiasValue: {value}')
                exposure_bias_value = str(value)
            elif('MeteringMode' in f'{tag}'):
                #print(f'MeteringMode: {value}')
                metering_mode = str(value)
            elif('Flash' in f'{tag}'): 
                #print(f'Flash: {value}')
                flash = str(value)
            elif('ExposureMode' in f'{tag}'):
                #print(f'ExposureMode: {value}')
                exposure_mode = str(value)
            elif('WhiteBalance' in f'{tag}'):
                #print(f'WhiteBalance: {value}')
                white_balance = str(value)
            elif('BracketMode' in f'{tag}'):
                #print(f'BracketMode: {value}')
                bracket_mode = str(value)
            elif('BracketValue' in f'{tag}'):
                #print(f'BracketValue: {value}')
                bracket_value = str(value)
            elif('BracketShotNumber' in f'{tag}'):
                #print(f'BracketShotNumber: {value}')
                bracket_shot_number = str(value)
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

        if exposure_time:
            lens_dict['exposure_time'] = exposure_time
        if fnumber:
            lens_dict['fnumber'] = fnumber
        if exposure_program:
            lens_dict['exposure_program'] = exposure_program
        if iso_speed:
            lens_dict['iso_speed'] = iso_speed
        if date_time_original:
            lens_dict['date_time_original'] = date_time_original
        if shutter_speed_value:
            lens_dict['shutter_speed_value'] = shutter_speed_value
        if exposure_bias_value:
            lens_dict['exposure_bias_value'] = exposure_bias_value
        if metering_mode:
            lens_dict['metering_mode'] = metering_mode
        if flash:
            lens_dict['flash'] = flash
        if exposure_mode:
            lens_dict['exposure_mode'] = exposure_mode
        if white_balance:
            lens_dict['white_balance'] = white_balance
        if bracket_mode:
            lens_dict['bracket_mode'] = bracket_mode
        if bracket_value:
            lens_dict['bracket_value'] = bracket_value
        if bracket_shot_number:
            lens_dict['bracket_shot_number'] = bracket_shot_number

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
        with rawpy.imread(rawfile) as raw:
            x_resolution, y_resolution = raw.raw_image_visible.shape[::-1]  # width, height



    except Exception:
        try:
            with rawpy.imread(rawfile) as raw:
                 xyz = raw.postprocess(
                    output_color=rawpy.ColorSpace.XYZ,
                    gamma=(1, 1),                # Linear
                    no_auto_bright=True,         # Preserve superbrights
                    output_bps=16,               # RawPy internal bit depth; final is float16
                    use_camera_wb=True           # Use in-camera white balance
                )
                 y_resolution, x_resolution = xyz.shape[:2]

        except Exception:

            try:
                img = cv2.imread(rawfile, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    x_resolution, y_resolution = img.shape[1], img.shape[0]
            except Exception:
                print(f"Error reading resolution from {rawfile}.")
                return None, None
        
    return x_resolution, y_resolution