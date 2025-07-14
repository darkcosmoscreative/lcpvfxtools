import os
import sys
import argparse
import pandas as pd

print("lcpvfxtools batch mode initialised")

# initialise config from module on path
current_dir = os.path.dirname(__file__)
print(f"Current directory: {current_dir}")
module_root = os.path.abspath(os.path.join(current_dir, '..'))
if module_root not in sys.path:
    sys.path.append(module_root)
    print(f"Added module root to sys.path: {module_root}")

from lcpvfxtools import config as cfg
from lcpvfxtools import cc_utils
from lcpvfxtools import exif_utils
from lcpvfxtools import db_utils

def process_batch(input_dir, bracketed=False, override_make=None, override_model=None, override_lens=None):
    output_dir = os.path.join(input_dir, "batch_exr")
    os.makedirs(output_dir, exist_ok=True)

    input_files = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
        and os.path.splitext(f)[1][1:].lower() in cfg.SUPPORTED_RAW_FORMATS
    ]

    # For bracketed, only generate maps for the first file
    maps_generated = False

    for fname in input_files:
        raw_path = os.path.join(input_dir, fname)
        basename, _ = os.path.splitext(fname)
        print(f"\nProcessing {fname}...")

        # Get lens/camera info and resolution
        lens_dict = exif_utils.get_camera_lens_dict(raw_path)
        if override_make:
            lens_dict['cam_maker'] = override_make
        if override_model:
            lens_dict['cam_model'] = override_model
        if override_lens:
            lens_dict['lens_model'] = override_lens
        xres, yres = exif_utils.get_resolution_from_exif(raw_path)

        # get best matching lens profile
        lens_db = db_utils.LensProfileDatabase(
            lcp_directory=cfg.LCP_DIR,
            pickle_file=cfg.PICKLE_FILE,
            force_reload=False
        )
        scores = db_utils.score_lens_profile(lens_dict, lens_db)
        if not scores:
            print(f"  No profile found for {fname}, skipping.")
            continue
        best_profile = scores[0]['profile'] if scores else None

        # 1. Create ACEScg EXR
        cc_utils.write_exr_from_cameraraw(
            write_dir=output_dir,
            basename=basename,
            raw_file_path=raw_path,
            lens_dict=lens_dict
        )

        # Only generate maps once if bracketed
        if bracketed and maps_generated:
            continue

        # fix for missing focal lengths in some profiles
        if not pd.isna(best_profile['FocalLengthX']):
            focal_length_x = best_profile['FocalLengthX']
        else:
            focal_length_x = 1.0
        if not pd.isna(best_profile['FocalLengthY']):
            focal_length_y = best_profile['FocalLengthY']
        else:
            focal_length_y = 1.0

        # 2. Create distortion/undistortion maps



        cc_utils.write_st_maps_from_params(
            write_dir=output_dir,
            basename=basename,
            x_resolution=xres,
            y_resolution=yres,
            focal_length_x=focal_length_x,
            focal_length_y=focal_length_y,
            radialdistortparam1=best_profile['RadialDistortParam1'],
            radialDistortparam2=best_profile['RadialDistortParam2'],
            radialDistortparam3=best_profile['RadialDistortParam3']
        )

        # 3. Create lens scores for vignette
        filtered_scores = db_utils.filter_profiles_by_best_combo(scores, lens_db)
        vignette_scores = db_utils.score_vignette_profiles(lens_dict, filtered_scores)
        best_vignette_profile = vignette_scores[0]['profile'] if vignette_scores else None

        if best_vignette_profile is not None:
            if not pd.isna(best_vignette_profile['FocalLengthX']):
                vfocal_legth_x = best_vignette_profile['FocalLengthX']
            else:
                vfocal_legth_x = 1.0
            if not pd.isna(best_vignette_profile['FocalLengthY']):
                vfocal_length_y = best_vignette_profile['FocalLengthY']
            else:
                vfocal_length_y = 1.0

            # 4. Create vignette map
            cc_utils.write_vignette_map_from_params(
                write_dir=output_dir,
                basename=basename,
                x_resolution=xres,
                y_resolution=yres,
                focal_length_x=vfocal_legth_x,
                focal_length_y=vfocal_length_y,
                vignette_param1=best_vignette_profile.get('VignetteModelParam1', 0.0),
                vignette_param2=best_vignette_profile.get('VignetteModelParam2', 0.0),
                vignette_param3=best_vignette_profile.get('VignetteModelParam3', 0.0)
            )

        # 5. Create lens scores for TCA
        tca_scores = db_utils.score_tca_profiles(lens_dict, filtered_scores)
        best_tca_profile = tca_scores[0]['profile'] if tca_scores else None
        if best_tca_profile is not None:
            if not pd.isna(best_tca_profile['FocalLengthX']):
                tca_focal_length_x = best_tca_profile['FocalLengthX']
            else:
                tca_focal_length_x = 1.0
            if not pd.isna(best_tca_profile['FocalLengthY']):
                tca_focal_length_y = best_tca_profile['FocalLengthY']
            else:
                tca_focal_length_y = 1.0

            # 6. Create TCA correction maps
            cc_utils.write_tca_maps_from_params(
                write_dir=output_dir,
                basename=basename,
                x_resolution=xres,
                y_resolution=yres,
                focal_length_x=tca_focal_length_x,
                focal_length_y=tca_focal_length_y,
                tca_redgreen_radial1=best_tca_profile.get('TCA_RedGreen_Radial1', 0.0),
                tca_redgreen_radial2=best_tca_profile.get('TCA_RedGreen_Radial2', 0.0),
                tca_redgreen_radial3=best_tca_profile.get('TCA_RedGreen_Radial3', 0.0),
                tca_green_radial1=best_tca_profile.get('TCA_Green_Radial1', 0.0),
                tca_green_radial2=best_tca_profile.get('TCA_Green_Radial2', 0.0),
                tca_green_radial3=best_tca_profile.get('TCA_Green_Radial3', 0.0),
                tca_bluegreen_radial1=best_tca_profile.get('TCA_BlueGreen_Radial1', 0.0),
                tca_bluegreen_radial2=best_tca_profile.get('TCA_BlueGreen_Radial2', 0.0),
                tca_bluegreen_radial3=best_tca_profile.get('TCA_BlueGreen_Radial3', 0.0)
            )

        if bracketed:
            maps_generated = True

        print(f"  Finished processing {fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert raw images to EXR and correction maps.")
    parser.add_argument("input_dir", help="Directory containing raw files to process")
    parser.add_argument("--bracketed", action="store_true", help="Only generate one set of maps for the folder (bracketed mode)")
    parser.add_argument("--make", type=str, help="Override camera make (EXIF)")
    parser.add_argument("--model", type=str, help="Override camera model (EXIF)")
    parser.add_argument("--lens", type=str, help="Override lens model (EXIF)")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a valid directory.")
        sys.exit(1)

    process_batch(
        args.input_dir,
        bracketed=args.bracketed,
        override_make=args.make,
        override_model=args.model,
        override_lens=args.lens
    )