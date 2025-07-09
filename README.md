# lcpvfxtools

**lcpvfxtools** is a Python toolkit for working with Adobe LCP (Lens Correction Profile) files and camera raw images. It provides a wxPython-based graphical UI for selecting camera/lens profiles, generating correction maps (distortion, vignette, TCA), and converting camera raw files to OpenEXR format in ACEScg colourspace. The core correction and conversion utilities can also be used from the command line or in your own scripts for batch processing.

**The short goals are:**
 - **DSLR camera raw development for a VFX pipeline**
 - **Ensuring colour accuracy**
 - **Providing nondestructive profile corrections for use in Nuke or other VFX softwares**

*Note: This toolkit is designed for rectilinear camera images only. Fisheye, panoramic, or strongly distorted wide-angle projections are not currently supported.*

---

## Table of Contents

- [Nuke Usage](#nuke-usage)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Launching the GUI](#launching-the-gui)
  - [Using the GUI](#using-the-gui)
- [Command-Line and Scripting Utilities](#command-line-and-scripting-utilities)
- [Modules Overview](#modules-overview)
- [License](#license)

---
## Nuke Usage
- Generate from interface:

<p align="center"><img width=400 src="https://github.com/user-attachments/assets/5ab0d4b4-58e1-4b27-9285-d2089c222b80"/></p>

- Apply in Nuke:

<p align="center"><img width=500 src="https://github.com/user-attachments/assets/606e9389-8e5c-4db6-a967-ea8624e9c590"/></p>

- Provided Example:

<p align="center"><img width=500 src="https://github.com/user-attachments/assets/9bda24ca-8a55-4801-8c6e-8aa2fb46e7f6"/></p?>

## Installation

1. **Clone the repository:**

This repository uses [Git LFS](https://git-lfs.github.com/) for large test files. Make sure Git LFS is installed **before cloning**:


   ```sh
   git clone https://github.com/darkcosmoscreative/lcpvfxtools.git
   cd lcpvfxtools
   ```


2. **Install dependencies:**

   Install requirements:

   ```sh
   pip install -r requirements.txt
   ```

   > **Note:** On macOS, you may need to install `wxPython` using a wheel or with Homebrew due to build requirements. See [wxPython downloads](https://wxpython.org/pages/downloads/) for details.

---


## Configuration

To set your system paths:

1. Copy the default config:
   ```sh
   cp lcpvfxtools/config/_default.py lcpvfxtools/config/_local_config.py


2. Edit `_local_config.py` to reflect your system's file paths.

    The application will use this file automatically. If `_local_config.py` is missing, it will fall back to placeholder defaults.

---

## Usage

### Launching the GUI

To start the graphical user interface:

```sh
python main.py
```

Or, if you have set up your environment, you can also run:

```sh
python -m lcpvfxtools.ui_utils
```

---

### Using the GUI

1. **Select a Camera Raw File:**  
   Click "Browse" and choose a supported raw file (`.cr2`, `.nef`, `.arw`, `.dng`).  
   > While newer extensions such as`.cr3` and `.HEIC` files
   > are not supported, Adobe's **Lightroom** can make non-
   > desctructive DNG copies of almost any photo type.
   > For users without Adobe toolsets, Adobe's Free DNG
   > converter will convert most RAW file types.
   > https://helpx.adobe.com/camera-raw/using/adobe-dng-converter.html

2. **Review and Select Camera/Lens Profile:**  
   The UI will auto-populate camera make, model, lens, and available focal/aperture/distances from the LCP database.

3. **Generate Correction Maps:**  
   - **Generate EXR:** Converts the raw file to OpenEXR (ACEScg color).
   - **Generate Distort/Undistort ST Maps:** Creates distortion correction maps.
   - **Generate Vignette Map:** Creates a vignette gain map.
   - **Generate TCA ST Maps:** Creates chromatic aberration correction maps.

4. **EXIF Debug Info:**  
   Expand the "Show EXIF Debug Info" pane to see all extracted EXIF metadata.

5. **Output:**  
   All generated files are saved in the same directory as your selected raw file, with descriptive filenames.

---

## Command-Line and Scripting Utilities

All core correction and conversion functions are available as Python functions in the cc_utils module.  
You can use these in your own scripts for batch processing.

### Example: Convert a RAW file to EXR

```python
from lcpvfxtools.cc_utils import write_exr_from_cameraraw

write_exr_from_cameraraw(
    write_dir="/path/to/output",
    basename="my_image",
    raw_file_path="/path/to/input.CR2"
)
```

### Example: Generate Distortion Maps

```python
from lcpvfxtools.cc_utils import write_st_maps_from_params

write_st_maps_from_params(
    write_dir="/path/to/output",
    basename="my_image",
    x_resolution=6000,
    y_resolution=4000,
    focal_length_x=0.99,
    focal_length_y=0.99,
    radialdistortparam1=0.01,
    radialDistortparam2=0.001,
    radialDistortparam3=0.0001
)
```

### Example: Batch Processing

You can chain these utilities in your own scripts to process multiple files, e.g.:

```python
import glob
from lcpvfxtools import exif_utils as exif_utils
from lcpvfxtools import cc_utils as cc_utils


for raw_path in glob.glob("/my/raws/*.CR2"):
   # extract a lens dict
   exif_data = exif_utils.get_camera_lens_dict(raw_path)

   basename = os.path.splitext(os.path.basename(raw_path))[0]
   cc_utils.write_exr_from_cameraraw("/my/output", basename, raw_path, exif_data)
```

---

## Modules Overview

- **cc_utils**: Core correction and conversion functions (distortion, vignette, TCA, EXR export).
- **db_utils**: LCP database parsing, profile matching, and interpolation.
- **exif_utils**: EXIF extraction from raw files.
- **ui_utils**: wxPython GUI for interactive use.
- **config**: Configuration and path management.


---

## License

MIT License (see `LICENSE` file for details).
