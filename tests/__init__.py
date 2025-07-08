import os
import sys
import shutil
import filecmp


print("lcpvfxtools.tests initialised")

# initialise config from module on path
current_dir = os.path.dirname(__file__)
module_root = os.path.abspath(os.path.join(current_dir, '../..'))
if module_root not in sys.path:
    sys.path.append(module_root)

from lcpvfxtools import config as cfg
from lcpvfxtools import cc_utils
from lcpvfxtools import exif_utils
from lcpvfxtools import db_utils

# from lcpvfxtools.cc_utils import (
#     write_exr_from_cameraraw,
#     write_st_maps_from_params,
#     write_vignette_map_from_params,
#     write_tca_maps_from_params,
# )
# from lcpvfxtools.exif_utils import get_camera_lens_dict, get_resolution_from_exif
# from lcpvfxtools import db_utils


INPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "input")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "output")

def clean_output_dir():
    """
    Remove all files in OUTPUT_DIR except .gitkeep.
    Keeps the directory and .gitkeep file intact.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        return

    for fname in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if fname == ".gitkeep":
            continue
        if os.path.isfile(fpath) or os.path.islink(fpath):
            os.remove(fpath)
        elif os.path.isdir(fpath):
            shutil.rmtree(fpath)

def process_all_inputs():
    """
    Process all raw image files in the input directory and generate all correction outputs.

    For each file in tests/data/input:
        - Reads the raw image and extracts lens/camera metadata and resolution.
        - Selects the best matching lens profile from the database.
        - Writes an ACEScg EXR to tests/data/output.
        - Writes distortion and undistortion ST maps to tests/data/output.
        - Writes a vignette map to tests/data/output.
        - Writes TCA (chromatic aberration) correction maps to tests/data/output.
    Output files are named consistently with the UI module's conventions.

    The output directory is cleaned (except for .gitkeep) before processing.
    """
    clean_output_dir()
    input_files = [
        f for f in os.listdir(INPUT_DIR)
        if os.path.isfile(os.path.join(INPUT_DIR, f))
    ]
    for fname in input_files:
        raw_path = os.path.join(INPUT_DIR, fname)
        basename, _ = os.path.splitext(fname)
        print(f"\nProcessing {fname}...")

        # Get lens/camera info and resolution
        lens_dict = exif_utils.get_camera_lens_dict(raw_path)
        xres, yres = exif_utils.get_resolution_from_exif(raw_path)

        # get best matching lens profile
        lens_db = db_utils.LensProfileDatabase(lcp_directory=cfg.LCP_DIR,
                                                pickle_file=cfg.PICKLE_FILE,
                                                force_reload=False)
        
        scores = db_utils.score_lens_profile(lens_dict, lens_db)
        if not scores:
            print(f"  No profile found for {fname}, skipping.")
            continue
        best_profile = scores[0]['profile']

        # 1. Create ACEScg EXR
        exr_path = cc_utils.write_exr_from_cameraraw(
            write_dir=OUTPUT_DIR,
            basename=basename,
            raw_file_path=raw_path,
            lens_dict=lens_dict
        )

        # 2. Create distortion/undistortion maps
        cc_utils.write_st_maps_from_params(
            write_dir=OUTPUT_DIR,
            basename=basename,
            x_resolution=xres,
            y_resolution=yres,
            focal_length_x=best_profile['FocalLengthX'],
            focal_length_y=best_profile['FocalLengthY'],
            radialdistortparam1=best_profile['RadialDistortParam1'],
            radialDistortparam2=best_profile['RadialDistortParam2'],
            radialDistortparam3=best_profile['RadialDistortParam3']
        )

        # 3. Create lens scores for vignette
        filtered_scores = db_utils.filter_profiles_by_best_combo(scores, lens_db)
        vignette_scores = db_utils.score_vignette_profiles(lens_dict, filtered_scores)
        best_vignette_profile = vignette_scores[0]['profile'] if vignette_scores else None


        # 4. Create vignette map
        cc_utils.write_vignette_map_from_params(
            write_dir=OUTPUT_DIR,
            basename=basename,
            x_resolution=xres,
            y_resolution=yres,
            focal_length_x=best_vignette_profile['FocalLengthX'],
            focal_length_y=best_vignette_profile['FocalLengthY'],
            vignette_param1=best_vignette_profile.get('VignetteModelParam1', 0.0),
            vignette_param2=best_vignette_profile.get('VignetteModelParam2', 0.0),
            vignette_param3=best_vignette_profile.get('VignetteModelParam3', 0.0)
        )

        # 5. Create lens scores for TCA
        filtered_scores = db_utils.filter_profiles_by_best_combo(scores, lens_db)
        tca_scores = db_utils.score_tca_profiles(lens_dict, filtered_scores)
        best_tca_profile = tca_scores[0]['profile'] if tca_scores else None


        # 6. Create TCA correction maps
        cc_utils.write_tca_maps_from_params(
            write_dir=OUTPUT_DIR,
            basename=basename,
            x_resolution=xres,
            y_resolution=yres,
            focal_length_x=best_tca_profile['FocalLengthX'],
            focal_length_y=best_tca_profile['FocalLengthY'],
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

        print(f"  Finished processing {fname}")

def compare_test_outputs():
    """
    Compare files in tests/data/output to those in tests/data/expected.
    Reports:
        - PASS: All expected files exist in output and contents match.
        - PARTIAL FAIL: All expected files exist, but some contents do not match.
        - FAIL: Some expected files are missing from output.
    """
    expected_dir = os.path.join(os.path.dirname(__file__), "data", "expected")
    output_dir = os.path.join(os.path.dirname(__file__), "data", "output")

    expected_files = sorted([
        f for f in os.listdir(expected_dir)
        if os.path.isfile(os.path.join(expected_dir, f))
    ])
    output_files = sorted([
        f for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f)) and f != ".gitkeep"
    ])

    missing_files = []
    mismatched_files = []
    matched_files = []

    for fname in expected_files:
        expected_path = os.path.join(expected_dir, fname)
        output_path = os.path.join(output_dir, fname)
        if not os.path.exists(output_path):
            missing_files.append(fname)
        else:
            # Use filecmp for binary comparison
            if filecmp.cmp(expected_path, output_path, shallow=False):
                matched_files.append(fname)
            else:
                mismatched_files.append(fname)

    # Reporting
    if missing_files:
        print("\nFAIL: The following expected files are missing from output:")
        for f in missing_files:
            print(f"  {f}")
        print("Test result: COMPLETE FAIL")
    elif mismatched_files:
        print("\nPARTIAL FAIL: The following files exist but do not match expected contents:")
        for f in mismatched_files:
            print(f"  {f}")
        print("Test result: PARTIAL FAIL")
    else:
        print("\nPASS: All files match expected outputs.")
        print("Test result: PASS")

    # print matched files
    if matched_files:
        print("\nMatched files:")
        for f in matched_files:
            print(f"  {f}")
    if mismatched_files:
        print("\nMismatched files:")
        for f in mismatched_files:
            print(f"  {f}")

if __name__ == "__main__":
    process_all_inputs()
    compare_test_outputs()