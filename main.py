import sys
import os
import wx

current_dir = os.path.dirname(__file__)
module_root = os.path.abspath(os.path.join(current_dir, '..'))

if module_root not in sys.path:
    sys.path.append(module_root)

if __name__ == "__main__":
    """
    Entry point for the LCP VFX Tools application.

    This script launches the wxPython GUI for camera raw profile selection and correction map generation.
    It sets up the Python path, imports the main UI class, and starts the wxPython event loop.
    """
    from lcpvfxtools.ui_utils import UIUtils

    app = wx.App(False)
    frame = UIUtils(None, title="LCP VFX Tools")
    app.MainLoop()