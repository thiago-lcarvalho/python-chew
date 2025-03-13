import cv2
import os

def extract_xml_content():
    """Extract the XML content from the cascade files and save it to files."""
    # Get the path to the OpenCV Haar cascade files
    cascade_path = os.path.dirname(cv2.__file__) + '/data/'
    
    # Create an output directory
    output_dir = 'xml_content'
    os.makedirs(output_dir, exist_ok=True)
    
    # List of cascade files we need
    cascade_files = [
        'haarcascade_frontalface_default.xml',
        'haarcascade_profileface.xml',
        'haarcascade_smile.xml'
    ]
    
    # Extract the content of each file
    for file in cascade_files:
        file_path = os.path.join(cascade_path, file)
        if os.path.exists(file_path):
            print(f"Extracting content from {file}...")
            
            # Read the file content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Generate the variable name
            var_name = file.replace('.xml', '').upper()
            
            # Save the content to a file
            output_file = os.path.join(output_dir, f"{var_name}.txt")
            with open(output_file, 'w') as f:
                f.write(f"{var_name}_XML = '''\n")
                f.write(content)
                f.write("\n'''\n")
            
            print(f"Saved content to {output_file}")
        else:
            print(f"File {file} not found at {file_path}")
    
    print(f"\nXML content has been saved to the '{output_dir}' directory.")
    print("You can copy the content from these files into the chewing_counter_gui_embedded.py file.")

if __name__ == "__main__":
    extract_xml_content() 