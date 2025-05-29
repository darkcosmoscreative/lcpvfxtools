# lcpvfxtools

**lcpvfxtools** is a Python toolkit for working with Adobe LCP (Lens Correction Profile) files and camera raw images. It provides a wxPython-based graphical user interface (GUI) for selecting camera/lens profiles, generating correction maps (distortion, vignette, TCA), and converting camera raw files to OpenEXR format. The core correction and conversion utilities can also be used from the command line or in your own scripts for batch processing.

**The short goals are:**
 - **DSLR camera raw development for a VFX pipeline**
 - **Ensuring colour accuracy**
 - **Providing profile corrections for use in Nuke or other VFX softwares**

---

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Launching the GUI](#launching-the-gui)
  - [Using the GUI](#using-the-gui)
- [Command-Line and Scripting Utilities](#command-line-and-scripting-utilities)
- [Modules Overview](#modules-overview)
- [License](#license)

---

## Installation

1. **Clone the repository:**

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
   > `.cr3` files are not supported.

2. **Review and Select Camera/Lens Profile:**  
   The UI will auto-populate camera make, model, lens, and available focal/aperture/distances from the LCP database.

3. **Generate Correction Maps:**  
   - **Generate EXR:** Converts the raw file to OpenEXR (XYZ color).
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
    focal_length_x=3000.0,
    focal_length_y=3000.0,
    radialdistortparam1=0.01,
    radialDistortparam2=0.001,
    radialDistortparam3=0.0001
)
```

### Example: Batch Processing

You can chain these utilities in your own scripts to process multiple files, e.g.:

```python
import glob
from lcpvfxtools.cc_utils import write_exr_from_cameraraw

for raw_path in glob.glob("/my/raws/*.CR2"):
    basename = os.path.splitext(os.path.basename(raw_path))[0]
    write_exr_from_cameraraw("/my/output", basename, raw_path)
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
