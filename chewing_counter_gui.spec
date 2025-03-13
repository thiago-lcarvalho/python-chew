# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Get the path to the OpenCV Haar cascade files
import cv2
cascade_path = os.path.dirname(cv2.__file__) + '/data/'

# Collect all the Haar cascade XML files
cascade_files = [(os.path.join(cascade_path, f), 'cv2/data/') 
                 for f in os.listdir(cascade_path) if f.endswith('.xml')]

a = Analysis(
    ['chewing_counter_gui.py'],
    pathex=[],
    binaries=[],
    datas=cascade_files,  # Include the Haar cascade XML files
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
    console=False,
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
