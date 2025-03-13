import cv2
import numpy as np
import pandas as pd
import argparse
import time
from datetime import datetime
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
import threading
import traceback
import sys
import tempfile

# This script embeds the cascade files directly to avoid issues with PyInstaller

class ChewingCounter:
    def __init__(self, threshold=0.015, view='front'):
        """Initialize the chewing counter with face detection."""
        # Create temporary files for the cascade classifiers
        self.temp_dir = tempfile.mkdtemp(prefix="chewing_counter_")
        print(f"Created temporary directory: {self.temp_dir}")
        
        # Create and load the cascade classifiers
        self.face_cascade = self.create_cascade('haarcascade_frontalface_default.xml', FRONTALFACE_DEFAULT_XML)
        self.profile_cascade = self.create_cascade('haarcascade_profileface.xml', PROFILEFACE_XML)
        self.mouth_cascade = self.create_cascade('haarcascade_smile.xml', SMILE_XML)
        
        self.threshold = threshold
        self.view = view
        self.prev_mouth_height = None
        self.prev_jaw_position = None
        self.chew_count = 0
        self.mouth_heights = []
        self.jaw_positions = []
        self.is_mouth_open = False
        self.timestamps = []
        self.last_chew_time = None
        self.min_chew_interval = 0.3  # Minimum time between chews in seconds
        
        # For side view detection
        self.prev_jaw_region = None
        self.jaw_movement_history = []
        self.side_threshold = 0.5  # Threshold for side view movement detection
    
    def __del__(self):
        """Clean up temporary files."""
        try:
            import shutil
            print(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Error cleaning up temporary files: {e}")
    
    def create_cascade(self, filename, xml_content):
        """Create a cascade classifier from embedded XML content."""
        try:
            # Create a temporary file with the XML content
            filepath = os.path.join(self.temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write(xml_content)
            
            print(f"Created temporary cascade file: {filepath}")
            
            # Load the cascade classifier
            classifier = cv2.CascadeClassifier(filepath)
            
            # Verify the classifier was loaded properly
            if classifier.empty():
                print(f"Failed to load classifier from {filepath} (empty classifier)")
                return cv2.CascadeClassifier()
            else:
                print(f"Successfully loaded cascade classifier from {filepath}")
                return classifier
        except Exception as e:
            print(f"Error creating cascade classifier: {e}")
            return cv2.CascadeClassifier()
        
    def process_frame(self, frame):
        """Process a single frame to detect chewing motion."""
        if self.view == 'front':
            return self._process_front_view(frame)
        else:
            return self._process_side_view(frame)
    
    # The rest of the class remains the same as in the original file
    # ...

# The rest of the code (ChewingCounterApp class, etc.) remains the same as in the original file
# ...

# Embedded XML content for the cascade classifiers
# These are truncated versions for demonstration - you'll need to replace them with the full XML content

# Frontal face cascade
FRONTALFACE_DEFAULT_XML = '''<?xml version="1.0"?>
<!--
    This is a truncated version of the haarcascade_frontalface_default.xml file.
    In a real implementation, you would include the entire XML content here.
-->
<opencv_storage>
<haarcascade_frontalface_default type_id="opencv-haar-classifier">
  <size>24 24</size>
  <stages>
    <_>
      <!-- stage 0 -->
      <trees>
        <_>
          <!-- tree 0 -->
          <_>
            <!-- root node -->
            <feature>
              <rects>
                <_>3 7 14 4 -1.</_>
                <_>3 9 14 2 2.</_>
              </rects>
              <tilted>0</tilted>
            </feature>
            <threshold>4.0141958743333817e-003</threshold>
            <left_val>0.0337941907346249</left_val>
            <right_val>0.8378106951713562</right_val>
          </_>
        </_>
      </trees>
      <stage_threshold>0.0337941907346249</stage_threshold>
      <parent>-1</parent>
      <next>-1</next>
    </_>
  </stages>
</haarcascade_frontalface_default>
</opencv_storage>
'''

# Profile face cascade
PROFILEFACE_XML = '''<?xml version="1.0"?>
<!--
    This is a truncated version of the haarcascade_profileface.xml file.
    In a real implementation, you would include the entire XML content here.
-->
<opencv_storage>
<haarcascade_profileface type_id="opencv-haar-classifier">
  <size>20 20</size>
  <stages>
    <_>
      <!-- stage 0 -->
      <trees>
        <_>
          <!-- tree 0 -->
          <_>
            <!-- root node -->
            <feature>
              <rects>
                <_>0 2 4 8 -1.</_>
                <_>2 2 2 8 2.</_>
              </rects>
              <tilted>0</tilted>
            </feature>
            <threshold>0.0322894908487797</threshold>
            <left_val>0.0698464289307594</left_val>
            <right_val>0.8226894736289978</right_val>
          </_>
        </_>
      </trees>
      <stage_threshold>0.0698464289307594</stage_threshold>
      <parent>-1</parent>
      <next>-1</next>
    </_>
  </stages>
</haarcascade_profileface>
</opencv_storage>
'''

# Smile cascade
SMILE_XML = '''<?xml version="1.0"?>
<!--
    This is a truncated version of the haarcascade_smile.xml file.
    In a real implementation, you would include the entire XML content here.
-->
<opencv_storage>
<haarcascade_smile type_id="opencv-haar-classifier">
  <size>20 20</size>
  <stages>
    <_>
      <!-- stage 0 -->
      <trees>
        <_>
          <!-- tree 0 -->
          <_>
            <!-- root node -->
            <feature>
              <rects>
                <_>0 8 20 12 -1.</_>
                <_>0 14 20 6 2.</_>
              </rects>
              <tilted>0</tilted>
            </feature>
            <threshold>0.0207667704671621</threshold>
            <left_val>0.0822292566299438</left_val>
            <right_val>0.6575316786766052</right_val>
          </_>
        </_>
      </trees>
      <stage_threshold>0.0822292566299438</stage_threshold>
      <parent>-1</parent>
      <next>-1</next>
    </_>
  </stages>
</haarcascade_smile>
</opencv_storage>
'''

# Add the rest of the original code here
class ChewingCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chewing Counter")
        self.root.geometry("400x200")
        
        self.video_path = None
        self.output_path = "chewing_data.xlsx"
        self.processing_thread = None
        
        self.create_widgets()

    def create_widgets(self):
        """Create the GUI widgets."""
        self.upload_button = tk.Button(self.root, text="Upload Video", command=self.upload_video)
        self.upload_button.pack(pady=10)
        
        self.progress_label = tk.Label(self.root, text="Progress:")
        self.progress_label.pack()
        
        self.progress_bar = Progressbar(self.root, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress_bar.pack(pady=10)
        
        self.start_button = tk.Button(self.root, text="Start Processing", command=self.start_processing, state=tk.DISABLED)
        self.start_button.pack(pady=10)
        
        self.status_label = tk.Label(self.root, text="")
        self.status_label.pack(pady=5)

    def upload_video(self):
        """Open a file dialog to select a video file."""
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if self.video_path:
            self.status_label.config(text=f"Selected: {os.path.basename(self.video_path)}")
            self.start_button.config(state=tk.NORMAL)

    def start_processing(self):
        """Start processing the video file in a separate thread."""
        if not self.video_path:
            messagebox.showerror("Error", "Please upload a video file first.")
            return
        
        # Disable the start button during processing
        self.start_button.config(state=tk.DISABLED)
        self.upload_button.config(state=tk.DISABLED)
        
        # Reset progress bar
        self.progress_bar['value'] = 0
        self.status_label.config(text="Processing video...")
        self.root.update_idletasks()
        
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(target=self.process_video_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def process_video_thread(self):
        """Process the video in a separate thread."""
        try:
            self.process_video(self.video_path, self.output_path)
        except Exception as e:
            # Use after() to schedule GUI updates from the thread
            self.root.after(0, lambda: self.show_error(str(e), traceback.format_exc()))
    
    def show_error(self, error_msg, traceback_info):
        """Show error message in the GUI."""
        print(f"Error: {error_msg}")
        print(f"Traceback: {traceback_info}")
        messagebox.showerror("Error", f"An error occurred: {error_msg}")
        self.status_label.config(text="Error occurred during processing")
        self.start_button.config(state=tk.NORMAL)
        self.upload_button.config(state=tk.NORMAL)

    def update_progress(self, value, status_text=None):
        """Update progress bar from the thread."""
        self.progress_bar['value'] = value
        if status_text:
            self.status_label.config(text=status_text)
        self.root.update_idletasks()

    def process_video(self, video_path, output_path):
        """Process the video file and count chewing movements."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror("Error", f"Could not open video file {video_path}"))
            self.root.after(0, lambda: self.status_label.config(text="Error: Could not open video file"))
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.upload_button.config(state=tk.NORMAL))
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        counter = ChewingCounter()
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % 3 == 0:  # Process every 3rd frame
                counter.process_frame(frame)
                
                # Update progress bar (use after() to schedule GUI updates from the thread)
                progress = (frame_idx / frame_count) * 100
                self.root.after(0, lambda p=progress: self.update_progress(p, f"Processing: {int(p)}%"))
            
            frame_idx += 1
        
        cap.release()
        
        # Save results
        try:
            counter.save_to_excel(output_path)
            self.root.after(0, lambda: self.update_progress(100, f"Complete! Detected {counter.chew_count} chews"))
            self.root.after(0, lambda: messagebox.showinfo("Processing Complete", 
                                                          f"Chewing data saved to {output_path}\nDetected {counter.chew_count} chews"))
        except Exception as e:
            self.root.after(0, lambda: self.show_error(f"Error saving results: {str(e)}", traceback.format_exc()))
        
        # Re-enable buttons
        self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.upload_button.config(state=tk.NORMAL))

if __name__ == "__main__":
    root = tk.Tk()
    app = ChewingCounterApp(root)
    root.mainloop() 