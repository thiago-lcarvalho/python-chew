# Chewing Counter - Windows Build Instructions

This document provides instructions for building the Chewing Counter application as a Windows executable.

## Prerequisites

1. Install Python 3.8 or later on your Windows machine.
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   ```
   venv\Scripts\activate
   ```
4. Install the required packages:
   ```
   pip install opencv-python numpy pandas openpyxl pyinstaller
   ```
   Note: `tkinter` is included with Python on Windows, so you don't need to install it separately.

## Building the Executable

1. Copy the following files to your Windows machine:
   - `chewing_counter_gui.py`
   - `chewing_counter_gui_windows.spec`
   - `extract_cascades.py`

2. Activate the virtual environment:
   ```
   venv\Scripts\activate
   ```

3. Extract the cascade files (IMPORTANT - do this first):
   ```
   python extract_cascades.py
   ```
   This will create a `cascades` directory with the necessary XML files.

4. Build the executable using the spec file:
   ```
   pyinstaller chewing_counter_gui_windows.spec
   ```

5. The executable will be created in the `dist` folder.

## Troubleshooting

If you encounter the error "assertion failed !empty in function cascadeclassifier detect multiscale", it means the Haar cascade XML files are not being found. Try the following:

1. Make sure you ran the `extract_cascades.py` script before building the executable.

2. Check that the `cascades` directory contains the XML files:
   ```
   dir cascades\*.xml
   ```

3. If the files are present but the error persists, try copying the XML files directly to the same directory as the executable:
   ```
   copy cascades\*.xml dist\
   ```

4. Run the executable with console output enabled to see detailed error messages:
   ```
   pyinstaller --console chewing_counter_gui.py
   ```

## Usage

1. Double-click the `chewing_counter_gui.exe` file to run the application.
2. Click "Upload Video" to select a video file.
3. Click "Start Processing" to analyze the video and count chewing movements.
4. The results will be saved to an Excel file in the same directory as the executable. 