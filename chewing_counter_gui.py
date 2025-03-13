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

class ChewingCounter:
    def __init__(self, threshold=0.015, view='front'):
        """Initialize the chewing counter with face detection."""
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load profile face detector for side view
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Load facial landmark detector (for mouth region)
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
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
        
    def process_frame(self, frame):
        """Process a single frame to detect chewing motion."""
        if self.view == 'front':
            return self._process_front_view(frame)
        else:
            return self._process_side_view(frame)
    
    def _process_front_view(self, frame):
        """Process frame for front view chewing detection."""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Process the largest face
            (x, y, w, h) = max(faces, key=lambda face: face[2] * face[3])
            
            # Define the lower half of the face (mouth region)
            mouth_region_y = y + int(h * 0.6)
            mouth_region_h = int(h * 0.4)
            mouth_region = gray[mouth_region_y:mouth_region_y + mouth_region_h, x:x + w]
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw mouth region
            cv2.rectangle(frame, (x, mouth_region_y), (x + w, mouth_region_y + mouth_region_h), (0, 255, 0), 2)
            
            # Calculate mouth height using edge detection and contours
            if mouth_region.size > 0:  # Check if mouth region is valid
                # Apply edge detection
                edges = cv2.Canny(mouth_region, 100, 200)
                
                # Find contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Get the largest contour (likely the mouth)
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Get bounding rectangle
                    _, _, _, mouth_height = cv2.boundingRect(largest_contour)
                    
                    # Normalize by face height
                    normalized_mouth_height = mouth_height / h
                    
                    # Store the mouth height
                    self.mouth_heights.append(normalized_mouth_height)
                    
                    # Detect chewing motion based on mouth height changes
                    if self.prev_mouth_height is not None:
                        # Check if mouth is opening
                        if normalized_mouth_height > self.prev_mouth_height + self.threshold and not self.is_mouth_open:
                            self.is_mouth_open = True
                        # Check if mouth is closing after being open
                        elif normalized_mouth_height < self.prev_mouth_height - self.threshold and self.is_mouth_open:
                            self.is_mouth_open = False
                            
                            # Check if enough time has passed since the last chew
                            current_time = datetime.now()
                            if (self.last_chew_time is None or 
                                (current_time - self.last_chew_time).total_seconds() > self.min_chew_interval):
                                self.chew_count += 1
                                self.timestamps.append(current_time)
                                self.last_chew_time = current_time
                    
                    self.prev_mouth_height = normalized_mouth_height
                    
                    # Display mouth height on the frame
                    cv2.putText(frame, f"Mouth: {normalized_mouth_height:.3f}", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display chew count on the frame
            cv2.putText(frame, f"Chews: {self.chew_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        return frame
    
    def _process_side_view(self, frame):
        """Process frame for side view chewing detection using a simpler approach."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try to detect profile faces first
        profile_faces = self.profile_cascade.detectMultiScale(
            blurred,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )
        
        # If no profile face detected, try regular face detection
        if len(profile_faces) == 0:
            profile_faces = self.face_cascade.detectMultiScale(
                blurred,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(30, 30)
            )
        
        if len(profile_faces) > 0:
            # Process the largest face
            (x, y, w, h) = max(profile_faces, key=lambda face: face[2] * face[3])
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Define jaw region (lower part of the face)
            jaw_region_y = y + int(h * 0.7)
            jaw_region_h = int(h * 0.3)
            jaw_region_x = x
            jaw_region_w = w
            
            # Draw jaw region
            cv2.rectangle(frame, (jaw_region_x, jaw_region_y), 
                         (jaw_region_x + jaw_region_w, jaw_region_y + jaw_region_h), 
                         (0, 255, 0), 2)
            
            # Extract jaw region
            jaw_region = blurred[jaw_region_y:jaw_region_y + jaw_region_h, 
                               jaw_region_x:jaw_region_x + jaw_region_w]
            
            if jaw_region.size > 0:
                # Calculate average intensity in the jaw region
                avg_intensity = np.mean(jaw_region)
                
                # Calculate vertical gradient (difference between top and bottom of jaw)
                if jaw_region.shape[0] > 2:  # Make sure there are enough rows
                    top_half = jaw_region[:jaw_region.shape[0]//2, :]
                    bottom_half = jaw_region[jaw_region.shape[0]//2:, :]
                    top_avg = np.mean(top_half)
                    bottom_avg = np.mean(bottom_half)
                    vertical_gradient = top_avg - bottom_avg
                    
                    # Store the jaw movement data
                    self.jaw_movement_history.append(vertical_gradient)
                    
                    # Detect chewing based on jaw movement pattern
                    if len(self.jaw_movement_history) > 5:
                        # Keep only the last 10 measurements
                        if len(self.jaw_movement_history) > 10:
                            self.jaw_movement_history.pop(0)
                        
                        # Calculate the variance of recent movements
                        recent_movements = self.jaw_movement_history[-5:]
                        movement_variance = np.var(recent_movements)
                        
                        # Check for alternating pattern (chewing)
                        # Calculate differences between consecutive measurements
                        diffs = np.diff(recent_movements)
                        sign_changes = np.sum(np.diff(np.signbit(diffs)))
                        
                        # If there are multiple sign changes and enough variance, it's likely chewing
                        if sign_changes >= 2 and movement_variance > self.side_threshold:
                            # Check if enough time has passed since the last chew
                            current_time = datetime.now()
                            if (self.last_chew_time is None or 
                                (current_time - self.last_chew_time).total_seconds() > self.min_chew_interval):
                                self.chew_count += 1
                                self.timestamps.append(current_time)
                                self.last_chew_time = current_time
                    
                    # Display jaw movement info
                    cv2.putText(frame, f"Jaw: {vertical_gradient:.3f}", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    if len(self.jaw_movement_history) > 5:
                        variance = np.var(self.jaw_movement_history[-5:])
                        cv2.putText(frame, f"Var: {variance:.3f}", (x, y - 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display chew count on the frame
            cv2.putText(frame, f"Chews: {self.chew_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Side View Mode", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        else:
            # Display message when no face detected
            cv2.putText(frame, "No profile face detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def save_to_excel(self, output_path):
        """Save chewing data to an Excel file."""
        if not self.timestamps:
            print("No chewing data to save.")
            return
        
        # Create a DataFrame with chewing data
        data = {
            'Timestamp': self.timestamps,
            'Chew Number': range(1, len(self.timestamps) + 1)
        }
        
        df = pd.DataFrame(data)
        
        # Calculate time differences between chews
        if len(self.timestamps) > 1:
            time_diffs = [(self.timestamps[i] - self.timestamps[i-1]).total_seconds() 
                          for i in range(1, len(self.timestamps))]
            time_diffs.insert(0, 0)  # First chew has no previous time
            df['Time Since Last Chew (s)'] = time_diffs
        
        # Save to Excel
        df.to_excel(output_path, index=False)
        print(f"Chewing data saved to {output_path}")

class ChewingCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chewing Counter")
        self.root.geometry("400x200")
        
        self.video_path = None
        self.output_path = "chewing_data.xlsx"
        
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

    def upload_video(self):
        """Open a file dialog to select a video file."""
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if self.video_path:
            messagebox.showinfo("Selected Video", f"Selected video: {os.path.basename(self.video_path)}")
            self.start_button.config(state=tk.NORMAL)

    def start_processing(self):
        """Start processing the video file."""
        if not self.video_path:
            messagebox.showerror("Error", "Please upload a video file first.")
            return
        
        self.progress_bar['value'] = 0
        self.root.update_idletasks()
        
        # Process the video
        self.process_video(self.video_path, self.output_path)

    def process_video(self, video_path, output_path):
        """Process the video file and count chewing movements."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            messagebox.showerror("Error", f"Could not open video file {video_path}")
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
            
            if frame_idx % 3 == 0:
                counter.process_frame(frame)
                
                # Update progress bar
                progress = (frame_idx / frame_count) * 100
                self.progress_bar['value'] = progress
                self.root.update_idletasks()
            
            frame_idx += 1
        
        cap.release()
        counter.save_to_excel(output_path)
        messagebox.showinfo("Processing Complete", f"Chewing data saved to {output_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChewingCounterApp(root)
    root.mainloop()