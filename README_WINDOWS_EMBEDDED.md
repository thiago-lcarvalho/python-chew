# Chewing Counter - Windows Build Instructions (Embedded Version)

This document provides instructions for building the embedded version of the Chewing Counter application as a Windows executable. This version has the cascade files embedded directly in the code, which should resolve the "assertion failed !empty in function cascadeclassifier detect multiscale" error.

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
   - `chewing_counter_gui_embedded.py` (This version has the cascade files embedded in the code)
   - `chewing_counter_gui_embedded.spec`

2. Activate the virtual environment:
   ```
   venv\Scripts\activate
   ```

3. Build the executable using the spec file:
   ```
   pyinstaller chewing_counter_gui_embedded.spec
   ```

4. The executable will be created in the `dist` folder.

## Troubleshooting

If you still encounter issues, try the following:

1. Run the executable with console output enabled to see detailed error messages:
   ```
   pyinstaller --console chewing_counter_gui_embedded.py
   ```

2. Check if the temporary files are being created correctly by looking at the console output.

3. If you need to modify the embedded XML content, you can use the `extract_xml_content.py` script to extract the full XML content from your OpenCV installation.

## Usage

1. Double-click the `chewing_counter_gui.exe` file to run the application.
2. Click "Upload Video" to select a video file.
3. Click "Start Processing" to analyze the video and count chewing movements.
4. The results will be saved to an Excel file in the same directory as the executable. 