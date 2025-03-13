# Chewing Counter

This Python script analyzes video footage to count the number of chewing movements a person makes and records the data to an Excel file. It supports both front-facing and side view videos.

## Features

- Supports both front and side view video analysis
- Detects faces using OpenCV's Haar Cascade Classifier
- Tracks mouth movements to identify chewing motions
- Uses optical flow tracking for subtle jaw movements in side view
- Records timestamps for each detected chew
- Exports data to Excel with timestamps and time between chews
- Provides visual feedback with face and mouth region visualization (optional)

## Requirements

- Python 3.7+ (Tested with Python 3.13)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository or download the files
2. Create a virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Basic usage with default parameters (front view):

```bash
python chewing_counter.py
```

For side view analysis:

```bash
python chewing_counter.py --view side
```

This will process the default video file (`Chew-1.mp4`) and save the results to `chewing_data.xlsx`.

### Command Line Arguments

- `--video`: Path to the video file (default: `Chew-1.mp4`)
- `--output`: Path to the output Excel file (default: `chewing_data.xlsx`)
- `--display`: Display the video with face detection while processing
- `--threshold`: Threshold for detecting chewing motion (default: 0.015)
- `--view`: View angle of the video, either 'front' or 'side' (default: 'front')

Example with custom parameters:

```bash
python chewing_counter.py --video my_video.mp4 --output results.xlsx --display --threshold 0.02 --view side
```

## How It Works

### Front View Mode

1. The script uses OpenCV's Haar Cascade Classifier to detect faces
2. It focuses on the lower part of the face (mouth region)
3. Edge detection and contour analysis are used to track mouth movements
4. Chewing is detected by analyzing the pattern of mouth opening and closing

### Side View Mode

1. The script uses a profile face detector to locate the face from the side
2. It focuses on the lower jaw region
3. Optical flow tracking is used to detect subtle jaw movements
4. Chewing is detected by analyzing patterns of vertical movement in the jaw region

### Data Export

The data is exported to an Excel file with the following columns:
- Timestamp: When the chew was detected
- Chew Number: Sequential number of the chew
- Time Since Last Chew (s): Time elapsed since the previous chew

## Choosing the Right Mode

- **Front View**: Better for videos where the subject is facing the camera directly
- **Side View**: Better for videos where the subject is in profile or at an angle
  - More sensitive to subtle jaw movements
  - Can detect chewing even when the mouth is not clearly visible

## Limitations

- Works best with clear, well-lit video of the subject
- Lighting conditions and video quality affect detection accuracy
- May not detect chews if the face is partially obscured
- Side view detection requires more stable video with less head movement

## Adjusting Sensitivity

If you're getting too many or too few detections, try adjusting the `--threshold` parameter:
- Increase the threshold to make detection less sensitive (fewer chews detected)
- Decrease the threshold to make detection more sensitive (more chews detected) 