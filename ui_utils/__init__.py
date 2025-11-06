import wx
import os
import sys
import math
import pandas as pd

# initialise config from module on path
current_dir = os.path.dirname(__file__)
module_root = os.path.abspath(os.path.join(current_dir, '../..'))
if module_root not in sys.path:
    sys.path.append(module_root)

from lcpvfxtools import config as cfg
from lcpvfxtools.db_utils import LensProfileDatabase
from lcpvfxtools import db_utils as db_utils
from lcpvfxtools import exif_utils as exif_utils
from lcpvfxtools import cc_utils as cc_utils

class UIUtils(wx.Frame):
    """
    Main wxPython UI class for the LCP VFX Tools application.
    Provides controls for selecting files, camera/lens options, and generating correction maps.
    """

    def __init__(self, parent, title):
        """
        Initialize the UIUtils frame and set up the interface.

        Args:
            parent (wx.Window): Parent window.
            title (str): Window title.
        """
        super().__init__(parent, title=title, size=(500, 700))

        # Store last EXIF data for debug display
        self.last_exif_data = {}

        # Initialize the UI
        self.init_ui()

        # check if pickle file exists
        if not os.path.exists(cfg.PICKLE_FILE):
            print("Pickle file not found, creating new lens profile database pickle...")
            force_reload = True
        else:
            print("Loading lens profile database from pickle...")
            force_reload = False

        # Use the LensProfileDatabase class from db_utils
        self.lens_db = LensProfileDatabase(
            lcp_directory=cfg.LCP_DIR,
            pickle_file=cfg.PICKLE_FILE,
            force_reload=force_reload
        )

        self.Show()

    def init_ui(self):
        """
        Set up the wxPython UI components and layout.
        """
        self.panel = wx.Panel(self)
        panel = self.panel
        self.vbox = wx.BoxSizer(wx.VERTICAL)

        # --- Source selection dropdown ---
        source_label = wx.StaticText(panel, label="Source:")
        self.vbox.Add(source_label, flag=wx.ALL, border=5)
        self.source_choices = ["From File", "Manual Entry"]
        self.source_dropdown = wx.ComboBox(panel, choices=self.source_choices, style=wx.CB_READONLY)
        self.source_dropdown.SetSelection(0)
        self.vbox.Add(self.source_dropdown, flag=wx.ALL, border=5)
        self.source_dropdown.Bind(wx.EVT_COMBOBOX, self.on_source_change)

        # --- Manual entry fields (hidden by default) ---
        self.manual_panel = wx.Panel(panel)
        manual_sizer = wx.BoxSizer(wx.VERTICAL)
        self.manual_directory_label = wx.StaticText(self.manual_panel, label="Output Directory:")
        self.manual_directory_button = wx.Button(self.manual_panel, label="Select Directory")
        self.manual_directory_button.Bind(wx.EVT_BUTTON, lambda event: self.select_write_directory())
        self.manual_name_label = wx.StaticText(self.manual_panel, label="File Basename:")
        self.manual_name_text = wx.TextCtrl(self.manual_panel, value="Custom_Basename")
        self.manual_xres_label = wx.StaticText(self.manual_panel, label="X Resolution:")
        self.manual_xres_text = wx.TextCtrl(self.manual_panel, value="4096")
        self.manual_yres_label = wx.StaticText(self.manual_panel, label="Y Resolution:")
        self.manual_yres_text = wx.TextCtrl(self.manual_panel, value="2160")
        manual_sizer.Add(self.manual_directory_label, flag=wx.ALL, border=2)
        manual_sizer.Add(self.manual_directory_button, flag=wx.EXPAND | wx.ALL, border=2)
        manual_sizer.Add(self.manual_name_label, flag=wx.ALL, border=2)
        manual_sizer.Add(self.manual_name_text, flag=wx.EXPAND | wx.ALL, border=2)
        manual_sizer.Add(self.manual_xres_label, flag=wx.ALL, border=2)
        manual_sizer.Add(self.manual_xres_text, flag=wx.EXPAND | wx.ALL, border=2)
        manual_sizer.Add(self.manual_yres_label, flag=wx.ALL, border=2)
        manual_sizer.Add(self.manual_yres_text, flag=wx.EXPAND | wx.ALL, border=2)
        self.manual_panel.SetSizer(manual_sizer)
        self.manual_panel.Hide()
        self.vbox.Add(self.manual_panel, flag=wx.EXPAND | wx.ALL, border=5)

        # --- File selection row: label, button, and filename display (hidden by default for manual mode) ---
        self.file_label = wx.StaticText(panel, label="Select Camera Raw File:")
        self.vbox.Add(self.file_label, flag=wx.ALL, border=5)

        self.file_button = wx.Button(panel, label="Browse")
        self.file_button.Bind(wx.EVT_BUTTON, self.select_file)
        self.vbox.Add(self.file_button, flag=wx.ALL, border=5)

        self.selected_file_label = wx.StaticText(panel, label="No file selected")
        self.vbox.Add(self.selected_file_label, flag=wx.ALL, border=5)

        self.reload_button = wx.Button(panel, label="Reload from File")
        self.reload_button.Disable()
        self.reload_button.Bind(wx.EVT_BUTTON, self.reload_from_file)
        self.vbox.Add(self.reload_button, flag=wx.ALL, border=5)

        self.vbox.Add(wx.StaticLine(panel), flag=wx.EXPAND | wx.ALL, border=10)


        # Camera Make dropdown
        self.camera_make_dropdown = wx.ComboBox(panel, style=wx.CB_READONLY)
        self.camera_make_dropdown.Disable()
        self.vbox.Add(wx.StaticText(panel, label="Camera Make:"), flag=wx.ALL, border=5)
        self.vbox.Add(self.camera_make_dropdown, flag=wx.EXPAND | wx.ALL, border=5)
        self.camera_make_dropdown.Bind(wx.EVT_COMBOBOX, self.on_camera_make_change)

        # Camera Model dropdown
        self.camera_model_dropdown = wx.ComboBox(panel, style=wx.CB_READONLY)
        self.camera_model_dropdown.Disable()
        self.vbox.Add(wx.StaticText(panel, label="Camera Model:"), flag=wx.ALL, border=5)
        self.vbox.Add(self.camera_model_dropdown, flag=wx.EXPAND | wx.ALL, border=5)
        self.camera_model_dropdown.Bind(wx.EVT_COMBOBOX, self.on_camera_model_change)

        # Lens Model dropdown
        self.lens_model_dropdown = wx.ComboBox(panel, style=wx.CB_READONLY)
        self.lens_model_dropdown.Disable()
        self.vbox.Add(wx.StaticText(panel, label="Lens Model:"), flag=wx.ALL, border=5)
        self.vbox.Add(self.lens_model_dropdown, flag=wx.EXPAND | wx.ALL, border=5)
        self.lens_model_dropdown.Bind(wx.EVT_COMBOBOX, self.on_lens_model_change)

        # Focal Length dropdown
        self.focal_dropdown = wx.ComboBox(panel, style=wx.CB_READONLY)
        self.focal_dropdown.Disable()
        self.vbox.Add(wx.StaticText(panel, label="Focal Length:"), flag=wx.ALL, border=5)
        self.vbox.Add(self.focal_dropdown, flag=wx.EXPAND | wx.ALL, border=5)

        # Focus Distance dropdown
        self.focus_dropdown = wx.ComboBox(panel, style=wx.CB_READONLY)
        self.focus_dropdown.Disable()
        self.vbox.Add(wx.StaticText(panel, label="Focus Distance:"), flag=wx.ALL, border=5)
        self.vbox.Add(self.focus_dropdown, flag=wx.EXPAND | wx.ALL, border=5)

        # Aperture dropdown
        self.aperture_dropdown = wx.ComboBox(panel, style=wx.CB_READONLY)
        self.aperture_dropdown.Disable()
        self.vbox.Add(wx.StaticText(panel, label="Aperture:"), flag=wx.ALL, border=5)
        self.vbox.Add(self.aperture_dropdown, flag=wx.EXPAND | wx.ALL, border=5)

        # Buttons
        # --- Generate EXR Button ---
        self.generate_exr_button = wx.Button(panel, label="Generate EXR")
        self.generate_exr_button.Disable()
        self.generate_exr_button.Bind(wx.EVT_BUTTON, self.generate_exr)
        self.vbox.Add(self.generate_exr_button, flag=wx.EXPAND | wx.ALL, border=5)

        # --- Divider line before distortion/vignette/TCA buttons ---
        self.vbox.Add(wx.StaticLine(panel), flag=wx.EXPAND | wx.ALL, border=10)
        # --- Distortion, Vignette, and TCA Buttons ---
        self.generate_distort_button = wx.Button(panel, label="Generate Distort/Undistort ST Maps")
        self.generate_distort_button.Disable()
        self.generate_distort_button.Bind(wx.EVT_BUTTON, self.generate_distort_maps)
        self.vbox.Add(self.generate_distort_button, flag=wx.EXPAND | wx.ALL, border=5)

        self.generate_vignette_button = wx.Button(panel, label="Generate Vignette Map")
        self.generate_vignette_button.Disable()
        self.generate_vignette_button.Bind(wx.EVT_BUTTON, self.generate_vignette_map)
        self.vbox.Add(self.generate_vignette_button, flag=wx.EXPAND | wx.ALL, border=5)

        self.generate_tca_button = wx.Button(panel, label="Generate TCA ST Maps")
        self.generate_tca_button.Disable()
        self.generate_tca_button.Bind(wx.EVT_BUTTON, self.generate_tca_maps)
        self.vbox.Add(self.generate_tca_button, flag=wx.EXPAND | wx.ALL, border=5)

        # --- EXIF Debug Section ---
        self.exif_debug_collapsible = wx.CollapsiblePane(panel, label="Show EXIF Debug Info", style=wx.CP_DEFAULT_STYLE)
        self.Bind(wx.EVT_COLLAPSIBLEPANE_CHANGED, self.on_exif_debug_toggle, self.exif_debug_collapsible)
        self.vbox.Add(self.exif_debug_collapsible, flag=wx.EXPAND | wx.ALL, border=5)

        self.exif_debug_text = wx.TextCtrl(self.exif_debug_collapsible.GetPane(), style=wx.TE_MULTILINE | wx.TE_READONLY)
        exif_debug_sizer = wx.BoxSizer(wx.VERTICAL)
        exif_debug_sizer.Add(self.exif_debug_text, 1, wx.EXPAND | wx.ALL, 2)
        self.exif_debug_collapsible.GetPane().SetSizer(exif_debug_sizer)
        self.exif_debug_collapsible.Collapse(True)

        panel.SetSizer(self.vbox)

        # Track current selections for filtering
        self.current_cam_make = None
        self.current_cam_model = None
        self.current_lens_model = None
        self.selected_file_path = None
        self.selected_file_exif = None

    def on_source_change(self, event):
        """
        Callback for when the source dropdown is changed.
        Shows/hides manual entry fields and file selection widgets, then triggers enable_ui.
        """
        self.update_source_visibility()
        self.Layout()
        if self.source_dropdown.GetValue() == "Manual Entry":
            self.populate_ui_for_manual_mode()
        else:
            self.enable_ui()

        # Force layout update
        self.vbox.Layout()
        self.vbox.Fit(self.panel)
        self.panel.Layout()
        self.panel.Fit()
        self.Layout()
        self.Fit()

    def update_source_visibility(self):
        """
        Show/hide manual entry and file selection widgets based on source selection.
        """
        mode = self.source_dropdown.GetValue()
        if mode == "Manual Entry":
            self.manual_panel.Show()
            self.file_label.Hide()
            self.file_button.Hide()
            self.selected_file_label.Hide()
            self.reload_button.Hide()
            self.generate_exr_button.Hide()
        else:
            self.manual_panel.Hide()
            self.file_label.Show()
            self.file_button.Show()
            self.selected_file_label.Show()
            self.reload_button.Show()
            self.generate_exr_button.Show()

    def on_exif_debug_toggle(self, event):
        """
        Callback for toggling the EXIF debug info pane.

        Args:
            event (wx.Event): The wxPython event object.
        """
        self.Layout()

    def update_exif_debug(self, exif_data):
        """
        Update the EXIF debug text control with formatted EXIF data.

        Args:
            exif_data (dict): EXIF data dictionary.
        """
        import json
        self.last_exif_data = exif_data
        pretty = json.dumps(exif_data, indent=2)
        self.exif_debug_text.SetValue(pretty)

    def select_file(self, event):
        """
        Open a file dialog for the user to select a camera raw file.
        Loads EXIF data and updates the UI accordingly.

        Args:
            event (wx.Event): The wxPython event object.
        """

        # use the global config for supported raw formats
        formats_wildcard = ""
        for one_ext in cfg.SUPPORTED_RAW_FORMATS:
            formats_wildcard += f"*.{str(one_ext.lower())};"
            formats_wildcard += f"*.{str(one_ext.upper())};"

        if formats_wildcard.endswith(";"):
            formats_wildcard = formats_wildcard[:-1]

        #print(formats_wildcard)

        with wx.FileDialog(
            self,
            "Select Camera Raw File",
            wildcard=f"Camera Raw Files ({formats_wildcard})|{formats_wildcard}|All Files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return  # User canceled the dialog

            file_path = file_dialog.GetPath()
            self.selected_file_path = file_path
            self.selected_file_label.SetLabel(os.path.basename(file_path))
            self.reload_button.Enable()

            exif_data = exif_utils.get_camera_lens_dict(file_path)
            self.selected_file_exif = exif_data  # Save for reload
            self.update_exif_debug(exif_data)

            scores = db_utils.score_lens_profile(exif_data, self.lens_db)
            '''
            for _ in range(10):
                print(f"Score {_}: {scores[_]}")
            '''

            self.current_cam_make = scores[0]['profile']['Make']
            self.current_cam_model = scores[0]['profile']['Model']
            self.current_lens_model = scores[0]['profile']['Lens']

            focal_length = exif_data.get('focal_length')
            focus_distance = exif_data.get('distance')
            aperture = exif_data.get('aperture')

            all_cam_makes, all_cam_models, all_lens_models = self.get_filtered_camera_lens_options(
                self.lens_db, selected_cam_make=self.current_cam_make, selected_cam_model=self.current_cam_model
            )
            self.enable_ui(
                all_cam_makes, all_cam_models, all_lens_models,
                self.current_cam_make, self.current_cam_model, self.current_lens_model,
                focal_length, focus_distance, aperture
            )

    def select_write_directory(self):
        """
        Prompt the user to select a directory for writing output files.
        Returns the selected directory path as a string, or None if cancelled.
        """
        with wx.DirDialog(self, "Select Output Directory", style=wx.DD_DEFAULT_STYLE) as dir_dialog:
            if dir_dialog.ShowModal() == wx.ID_CANCEL:
                return   # User cancelled
            
            self.manual_write_dir = dir_dialog.GetPath()


    def populate_ui_for_manual_mode(self):
        """
        Populate the UI dropdowns for manual mode using the first available profile in the database.
        """
        if self.lens_db is None or not hasattr(self.lens_db, 'data') or self.lens_db.data is None:
            return

        df = self.lens_db.data
        if len(df) == 0:
            return

        # Get the first row as a starting point
        first_profile = df.iloc[0]
        cam_make = first_profile['Make']
        cam_model = first_profile['Model']
        lens_model = first_profile['Lens']
        focal_length = first_profile.get('FocalLength', None)
        focus_distance = first_profile.get('FocusDistance', None)
        aperture = first_profile.get('ApertureValue', None)

        all_cam_makes, all_cam_models, all_lens_models = self.get_filtered_camera_lens_options(
            self.lens_db, selected_cam_make=cam_make, selected_cam_model=cam_model
        )

        self.enable_ui(
            all_cam_makes, all_cam_models, all_lens_models,
            cam_make, cam_model, lens_model,
            focal_length, focus_distance, aperture
        )
            
    def reload_from_file(self, event):
        """
        Reload the UI and EXIF debug info from the last selected file.

        Args:
            event (wx.Event): The wxPython event object.
        """
        if not self.selected_file_path or not self.selected_file_exif:
            return
        exif_data = self.selected_file_exif
        self.update_exif_debug(exif_data)

        scores = db_utils.score_lens_profile(exif_data, self.lens_db)
        '''
        for _ in range(10):
            print(f"Score {_}: {scores[_]}")
        '''

        self.current_cam_make = scores[0]['profile']['Make']
        self.current_cam_model = scores[0]['profile']['Model']
        self.current_lens_model = scores[0]['profile']['Lens']

        focal_length = exif_data.get('focal_length')
        focus_distance = exif_data.get('distance')
        aperture = exif_data.get('aperture')

        all_cam_makes, all_cam_models, all_lens_models = self.get_filtered_camera_lens_options(
            self.lens_db, selected_cam_make=self.current_cam_make, selected_cam_model=self.current_cam_model
        )
        self.enable_ui(
            all_cam_makes, all_cam_models, all_lens_models,
            self.current_cam_make, self.current_cam_model, self.current_lens_model,
            focal_length, focus_distance, aperture
        )

    def get_filtered_camera_lens_options(self, db, selected_cam_make=None, selected_cam_model=None):
        """
        Get available camera makes, models, and lens models from the database,
        optionally filtered by selected make/model.

        Args:
            db (LensProfileDatabase): The lens profile database.
            selected_cam_make (str, optional): Selected camera make.
            selected_cam_model (str, optional): Selected camera model.

        Returns:
            tuple: (cam_makes, cam_models, lens_models) as lists.
        """
        if db is None or not hasattr(db, 'data') or db.data is None:
            return [], [], []
        df = db.data

        # Camera makes: always all available
        cam_makes = df['Make'].dropna().unique().tolist()

        # Camera models: filter by make if selected
        if selected_cam_make:
            cam_models = df[df['Make'] == selected_cam_make]['Model'].dropna().unique().tolist()
        else:
            cam_models = df['Model'].dropna().unique().tolist()

        # Lenses: filter by make/model if selected
        lens_rows = df
        if selected_cam_make:
            lens_rows = lens_rows[lens_rows['Make'] == selected_cam_make]
        if selected_cam_model:
            lens_rows = lens_rows[lens_rows['Model'] == selected_cam_model]

        lens_models = lens_rows['Lens'].dropna().unique().tolist()

        return cam_makes, cam_models, lens_models

    def get_available_focal_focus_aperture(self, cam_make, cam_model, lens_model):
        """
        Get available focal lengths, focus distances, and apertures for a given camera/lens.

        Args:
            cam_make (str): Camera make.
            cam_model (str): Camera model.
            lens_model (str): Lens model.

        Returns:
            tuple: (focal_lengths, focus_distances, apertures) as sorted lists.
        """
        df = self.lens_db.data
        if cam_model != '':
            filtered = df[
                (df['Make'] == cam_make) &
                (df['Model'] == cam_model) &
                (df['Lens'] == lens_model)
            ]
        else:
            filtered = df[
                        (df['Make'] == cam_make) &
                        (df['Lens'] == lens_model)
            ]

        focal_lengths = sorted(filtered['FocalLength'].dropna().unique().tolist())
        focus_distances = sorted(filtered['FocusDistance'].dropna().unique().tolist())
        apertures = sorted(filtered['ApertureValue'].dropna().unique().tolist())
        return focal_lengths, focus_distances, apertures
    
    def get_lens_dict_from_interface(self):
        """
        Get the selected camera/lens/focal/aperture/distance values from the UI.

        Returns:
            dict: Dictionary with keys 'cam_maker', 'cam_model', 'lens_model', 'focal_length', 'distance', 'aperture'.
        """
        cam_make = self.camera_make_dropdown.GetValue()
        cam_model = self.camera_model_dropdown.GetValue()
        lens_model = self.lens_model_dropdown.GetValue()
        focal_length = self.focal_dropdown.GetValue()
        focus_distance = self.focus_dropdown.GetValue()
        aperture = self.aperture_dropdown.GetValue()

        lens_dict = {
            'cam_maker': cam_make,
            'cam_model': cam_model,
            'lens_model': lens_model,
            'focal_length': float(focal_length),
            'distance': float(focus_distance),
            'aperture': float(aperture)
        }
        return lens_dict

    def enable_ui(self, all_cam_makes=None, all_cam_models=None, all_lens_models=None,
                  best_cam_make=None, best_cam_model=None, best_lens_model=None,
                  focal_length=None, focus_distance=None, aperture=None):
        """
        Enable and populate the UI dropdowns and buttons with available options.

        Args:
            all_cam_makes (list): All available camera makes.
            all_cam_models (list): All available camera models.
            all_lens_models (list): All available lens models.
            best_cam_make (str): Best matching camera make.
            best_cam_model (str): Best matching camera model.
            best_lens_model (str): Best matching lens model.
            focal_length (float): Focal length value.
            focus_distance (float): Focus distance value.
            aperture (float): Aperture value.
        """
        # Enable dropdowns and buttons
        self.camera_make_dropdown.Enable()
        self.camera_model_dropdown.Enable()
        self.lens_model_dropdown.Enable()
        self.focal_dropdown.Enable()
        self.focus_dropdown.Enable()
        self.aperture_dropdown.Enable()
        self.generate_exr_button.Enable()
        self.generate_distort_button.Enable()
        self.generate_vignette_button.Enable()
        self.generate_tca_button.Enable()

        # Populate and set camera make dropdown
        if all_cam_makes:
            self.camera_make_dropdown.SetItems(all_cam_makes)
            if best_cam_make and best_cam_make in all_cam_makes:
                self.camera_make_dropdown.SetValue(best_cam_make)
            elif all_cam_makes:
                self.camera_make_dropdown.SetSelection(0)

        # Populate and set camera model dropdown
        if all_cam_models:
            self.camera_model_dropdown.SetItems(all_cam_models)
            if best_cam_model and best_cam_model in all_cam_models:
                self.camera_model_dropdown.SetValue(best_cam_model)
            elif all_cam_models:
                self.camera_model_dropdown.SetSelection(0)

        # Populate and set lens model dropdown
        if all_lens_models:
            self.lens_model_dropdown.SetItems(all_lens_models)
            if best_lens_model and best_lens_model in all_lens_models:
                self.lens_model_dropdown.SetValue(best_lens_model)
            elif all_lens_models:
                self.lens_model_dropdown.SetSelection(0)

        # Populate and set focal length dropdown
        if focal_length is not None:
            self.focal_dropdown.SetItems([str(focal_length)])
            self.focal_dropdown.SetValue(str(focal_length))
        else:
            self.focal_dropdown.SetItems([])
            self.focal_dropdown.SetValue("")

        # Populate and set focus distance dropdown
        if focus_distance is not None:
            self.focus_dropdown.SetItems([str(focus_distance)])
            self.focus_dropdown.SetValue(str(focus_distance))
        else:
            self.focus_dropdown.SetItems([])
            self.focus_dropdown.SetValue("")

        # Populate and set aperture dropdown
        if aperture is not None:
            self.aperture_dropdown.SetItems([str(aperture)])
            self.aperture_dropdown.SetValue(str(aperture))
        else:
            self.aperture_dropdown.SetItems([])
            self.aperture_dropdown.SetValue("")

    # --- UI Callbacks for Filtering ---

    def on_camera_make_change(self, event):
        """
        Callback for when the camera make dropdown is changed.

        Args:
            event (wx.Event): The wxPython event object.
        """
        selected_make = self.camera_make_dropdown.GetValue()
        self.current_cam_make = selected_make
        # Reset model and lens selections
        self.current_cam_model = None
        self.current_lens_model = None

        all_cam_makes, all_cam_models, all_lens_models = self.get_filtered_camera_lens_options(
            self.lens_db, selected_cam_make=selected_make
        )
        self.enable_ui(all_cam_makes, all_cam_models, all_lens_models,
                       selected_make, None, None, None, None, None)

    def on_camera_model_change(self, event):
        """
        Callback for when the camera model dropdown is changed.

        Args:
            event (wx.Event): The wxPython event object.
        """
        selected_make = self.camera_make_dropdown.GetValue()
        selected_model = self.camera_model_dropdown.GetValue()
        self.current_cam_make = selected_make
        self.current_cam_model = selected_model
        self.current_lens_model = None

        all_cam_makes, all_cam_models, all_lens_models = self.get_filtered_camera_lens_options(
            self.lens_db, selected_cam_make=selected_make, selected_cam_model=selected_model
        )
        self.enable_ui(all_cam_makes, all_cam_models, all_lens_models,
                       selected_make, selected_model, None, None, None, None)

    def on_lens_model_change(self, event):
        """
        Callback for when the lens model dropdown is changed.

        Args:
            event (wx.Event): The wxPython event object.
        """
        selected_make = self.camera_make_dropdown.GetValue()
        selected_model = self.camera_model_dropdown.GetValue()
        selected_lens = self.lens_model_dropdown.GetValue()
        self.current_cam_make = selected_make
        self.current_cam_model = selected_model
        self.current_lens_model = selected_lens

        # Get available values for this combo
        focal_lengths, focus_distances, apertures = self.get_available_focal_focus_aperture(
            selected_make, selected_model, selected_lens
        )

        # Update the dropdowns
        self.focal_dropdown.SetItems([str(f) for f in focal_lengths])
        self.focus_dropdown.SetItems([str(d) for d in focus_distances])
        self.aperture_dropdown.SetItems([str(a) for a in apertures])

        # Optionally set default values
        if focal_lengths:
            self.focal_dropdown.SetValue(str(focal_lengths[0]))
        if focus_distances:
            self.focus_dropdown.SetValue(str(focus_distances[0]))
        if apertures:
            self.aperture_dropdown.SetValue(str(apertures[0]))

    def generate_exr(self, event):
        """
        Generate an EXR file from the selected camera raw file and show a dialog on completion.

        Args:
            event (wx.Event): The wxPython event object.
        """
        interface_lens_dict = self.get_lens_dict_from_interface()

        if self.selected_file_path:
            file_dir = os.path.dirname(self.selected_file_path)
            file_name = os.path.basename(self.selected_file_path)
            file_basename = os.path.splitext(file_name)[0]

        if os.path.exists(self.selected_file_path):

            # Generate the EXR file
            cc_utils.write_exr_from_cameraraw(
                write_dir=file_dir,
                basename=file_basename,
                raw_file_path=self.selected_file_path,
                lens_dict=interface_lens_dict

            )
            dlg = wx.MessageDialog(self, "EXR file generated successfully.", "Success", wx.OK | wx.ICON_INFORMATION)
            dlg.ShowModal()
            dlg.Destroy()


    def generate_distort_maps(self, event):
        """
        Generate distortion and undistortion ST maps and show a dialog on completion.
        """
        interface_lens_dict = self.get_lens_dict_from_interface()
        scores = db_utils.score_lens_profile(interface_lens_dict, self.lens_db)

        # Determine mode
        mode = self.source_dropdown.GetValue()
        if mode == "Manual Entry":
            file_dir = getattr(self, "manual_write_dir", None)
            file_basename = self.manual_name_text.GetValue()
            try:
                x_resolution = int(self.manual_xres_text.GetValue())
                y_resolution = int(self.manual_yres_text.GetValue())
            except Exception:
                dlg = wx.MessageDialog(self, "Invalid resolution values.", "Error", wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
                return
        else:
            if not self.selected_file_path:
                dlg = wx.MessageDialog(self, "No file selected.", "Error", wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
                return
            file_dir = os.path.dirname(self.selected_file_path)
            file_name = os.path.basename(self.selected_file_path)
            file_basename = os.path.splitext(file_name)[0]
            x_resolution, y_resolution = exif_utils.get_resolution_from_exif(self.selected_file_path)

        best_profile = scores[0]['profile']
        if not pd.isna(best_profile['FocalLengthX']):
            focal_length_x = best_profile['FocalLengthX']
        else:
            focal_length_x = 1.0
        if not pd.isna(best_profile['FocalLengthY']):
            focal_length_y = best_profile['FocalLengthY']
        else:
            focal_length_y = 1.0
        radial_distort_param1 = best_profile['RadialDistortParam1']
        radial_distort_param2 = best_profile['RadialDistortParam2']
        radial_distort_param3 = best_profile['RadialDistortParam3']
        if isinstance(radial_distort_param1, float):
            cc_utils.write_st_maps_from_params(
                write_dir=file_dir,
                basename=file_basename,
                x_resolution=x_resolution,
                y_resolution=y_resolution,
                focal_length_x=focal_length_x,
                focal_length_y=focal_length_y,
                radialdistortparam1=radial_distort_param1,
                radialDistortparam2=radial_distort_param2,
                radialDistortparam3=radial_distort_param3
            )
            dlg = wx.MessageDialog(self, "ST maps generated successfully.", "Success", wx.OK | wx.ICON_INFORMATION)
            dlg.ShowModal()
            dlg.Destroy()

    def generate_vignette_map(self, event):
        """
        Generate a vignette gain map and show a dialog on completion.
        """
        interface_lens_dict = self.get_lens_dict_from_interface()
        lens_scores = db_utils.score_lens_profile(interface_lens_dict, self.lens_db)
        filtered_scores = db_utils.filter_profiles_by_best_combo(lens_scores, self.lens_db)
        scores = db_utils.score_vignette_profiles(interface_lens_dict, filtered_scores)

        mode = self.source_dropdown.GetValue()
        if mode == "Manual Entry":
            file_dir = getattr(self, "manual_write_dir", None)
            file_basename = self.manual_name_text.GetValue()
            try:
                x_resolution = int(self.manual_xres_text.GetValue())
                y_resolution = int(self.manual_yres_text.GetValue())
            except Exception:
                dlg = wx.MessageDialog(self, "Invalid resolution values.", "Error", wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
                return
        else:
            if not self.selected_file_path:
                dlg = wx.MessageDialog(self, "No file selected.", "Error", wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
                return
            file_dir = os.path.dirname(self.selected_file_path)
            file_name = os.path.basename(self.selected_file_path)
            file_basename = os.path.splitext(file_name)[0]
            x_resolution, y_resolution = exif_utils.get_resolution_from_exif(self.selected_file_path)

        if scores != []:
            best_profile = scores[0]['profile']
            if not pd.isna(best_profile['FocalLengthX']):
                focal_length_x = best_profile['FocalLengthX']
            else:
                focal_length_x = 1.0
            if not pd.isna(best_profile['FocalLengthY']):
                focal_length_y = best_profile['FocalLengthY']
            else:
                focal_length_y = 1.0

            vignette_param1 = best_profile['VignetteModelParam1']
            vignette_param2 = best_profile['VignetteModelParam2']
            vignette_param3 = best_profile['VignetteModelParam3']
            if isinstance(vignette_param1, float) and not math.isnan(vignette_param1):
                cc_utils.write_vignette_map_from_params(
                    write_dir=file_dir,
                    basename=file_basename,
                    x_resolution=x_resolution,
                    y_resolution=y_resolution,
                    focal_length_x=focal_length_x,
                    focal_length_y=focal_length_y,
                    vignette_param1=vignette_param1,
                    vignette_param2=vignette_param2,
                    vignette_param3=vignette_param3
                )
                dlg = wx.MessageDialog(self, "Vignette map generated successfully.", "Success", wx.OK | wx.ICON_INFORMATION)
                dlg.ShowModal()
                dlg.Destroy()
            else:
                dlg = wx.MessageDialog(self, "Profile does not have the required data for vignette map.", "Information", wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
        else:
            dlg = wx.MessageDialog(self, "Profile does not have the required data for vignette map.", "Information", wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()

    def generate_tca_maps(self, event):
        """
        Generate TCA (transverse chromatic aberration) maps and show a dialog on completion.
        """
        interface_lens_dict = self.get_lens_dict_from_interface()
        lens_scores = db_utils.score_lens_profile(interface_lens_dict, self.lens_db)
        filtered_scores = db_utils.filter_profiles_by_best_combo(lens_scores, self.lens_db)
        scores = db_utils.score_tca_profiles(interface_lens_dict, filtered_scores)

        mode = self.source_dropdown.GetValue()
        if mode == "Manual Entry":
            file_dir = getattr(self, "manual_write_dir", None)
            file_basename = self.manual_name_text.GetValue()
            try:
                x_resolution = int(self.manual_xres_text.GetValue())
                y_resolution = int(self.manual_yres_text.GetValue())
            except Exception:
                dlg = wx.MessageDialog(self, "Invalid resolution values.", "Error", wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
                return
        else:
            if not self.selected_file_path:
                dlg = wx.MessageDialog(self, "No file selected.", "Error", wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
                return
            file_dir = os.path.dirname(self.selected_file_path)
            file_name = os.path.basename(self.selected_file_path)
            file_basename = os.path.splitext(file_name)[0]
            x_resolution, y_resolution = exif_utils.get_resolution_from_exif(self.selected_file_path)

        if scores != []:
            best_profile = scores[0]['profile']
            if not pd.isna(best_profile['FocalLengthX']):
                focal_length_x = best_profile['FocalLengthX']
            else:
                focal_length_x = 1.0
            if not pd.isna(best_profile['FocalLengthY']):
                focal_length_y = best_profile['FocalLengthY']
            else:
                focal_length_y = 1.0

            tca_redgreen_radial1 = best_profile['TCA_RedGreen_Radial1']
            tca_redgreen_radial2 = best_profile['TCA_RedGreen_Radial2']
            tca_redgreen_radial3 = best_profile['TCA_RedGreen_Radial3']
            tca_green_radial1 = best_profile['TCA_Green_Radial1']
            tca_green_radial2 = best_profile['TCA_Green_Radial2']
            tca_green_radial3 = best_profile['TCA_Green_Radial3']
            tca_bluegreen_radial1 = best_profile['TCA_BlueGreen_Radial1']
            tca_bluegreen_radial2 = best_profile['TCA_BlueGreen_Radial2']
            tca_bluegreen_radial3 = best_profile['TCA_BlueGreen_Radial3']
            if isinstance(tca_redgreen_radial1, float) and not math.isnan(tca_redgreen_radial1):
                cc_utils.write_tca_maps_from_params(
                    write_dir=file_dir,
                    basename=file_basename,
                    x_resolution=x_resolution,
                    y_resolution=y_resolution,
                    focal_length_x=focal_length_x,
                    focal_length_y=focal_length_y,
                    tca_redgreen_radial1=tca_redgreen_radial1,
                    tca_redgreen_radial2=tca_redgreen_radial2,
                    tca_redgreen_radial3=tca_redgreen_radial3,
                    tca_green_radial1=tca_green_radial1,
                    tca_green_radial2=tca_green_radial2,
                    tca_green_radial3=tca_green_radial3,
                    tca_bluegreen_radial1=tca_bluegreen_radial1,
                    tca_bluegreen_radial2=tca_bluegreen_radial2,
                    tca_bluegreen_radial3=tca_bluegreen_radial3
                )
                dlg = wx.MessageDialog(self, "TCA maps generated successfully.", "Success", wx.OK | wx.ICON_INFORMATION)
                dlg.ShowModal()
                dlg.Destroy()
            else:
                dlg = wx.MessageDialog(self, "Profile does not have the required data for TCA maps.", "Information", wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
        else:
            dlg = wx.MessageDialog(self, "Profile does not have the required data for TCA maps.", "Information", wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()


if __name__ == "__main__":

    app = wx.App(False)
    frame = UIUtils(None, title="LCP VFX Tools")

    # Add a File menu with Quit (for Cmd+Q on Mac)
    menubar = wx.MenuBar()
    file_menu = wx.Menu()
    quit_item = file_menu.Append(wx.ID_EXIT, "Quit\tCtrl+Q" if sys.platform != "darwin" else "Quit\tCmd+Q")
    menubar.Append(file_menu, "&File")
    frame.SetMenuBar(menubar)
    frame.Bind(wx.EVT_MENU, lambda evt: frame.Close(), quit_item)

    app.MainLoop()