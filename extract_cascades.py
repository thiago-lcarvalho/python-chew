import cv2
import os
import shutil

def extract_cascade_files():
    """Extract OpenCV cascade files to a local directory."""
    print("Extracting OpenCV cascade files...")
    
    # Create cascades directory if it doesn't exist
    os.makedirs('cascades', exist_ok=True)
    
    # Get the path to the OpenCV Haar cascade files
    cascade_path = os.path.dirname(cv2.__file__) + '/data/'
    print(f"OpenCV cascade path: {cascade_path}")
    
    # Copy all XML files to the cascades directory
    count = 0
    for file in os.listdir(cascade_path):
        if file.endswith('.xml'):
            src = os.path.join(cascade_path, file)
            dst = os.path.join('cascades', file)
            shutil.copy(src, dst)
            print(f"Copied {file} to cascades directory")
            count += 1
    
    print(f"Extracted {count} cascade files to the 'cascades' directory")

if __name__ == "__main__":
    extract_cascade_files() 