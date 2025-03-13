# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Get the path to the OpenCV Haar cascade files
import cv2
cascade_path = os.path.dirname(cv2.__file__) + '/data/'

# Collect all the Haar cascade XML files from OpenCV
cascade_files = [(os.path.join(cascade_path, f), 'cv2/data/') 
                 for f in os.listdir(cascade_path) if f.endswith('.xml')]

# Include the local cascades directory if it exists
local_cascade_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cascades')
if os.path.exists(local_cascade_dir):
    local_cascade_files = [(os.path.join(local_cascade_dir, f), 'cascades/') 
                          for f in os.listdir(local_cascade_dir) if f.endswith('.xml')]
    # Also copy to root directory for easier access
    root_cascade_files = [(os.path.join(local_cascade_dir, f), './') 
                         for f in os.listdir(local_cascade_dir) if f.endswith('.xml')]
    all_data_files = cascade_files + local_cascade_files + root_cascade_files
else:
    # For Windows, also copy to root directory for easier access
    windows_cascade_files = [(os.path.join(cascade_path, f), './') 
                            for f in os.listdir(cascade_path) if f.endswith('.xml')]
    all_data_files = cascade_files + windows_cascade_files

a = Analysis(
    ['chewing_counter_gui.py'],
    pathex=[],
    binaries=[],
    datas=all_data_files,  # Include the Haar cascade XML files
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='chewing_counter_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# For macOS, create an app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='chewing_counter_gui.app',
        icon=None,
        bundle_identifier=None,
    ) 