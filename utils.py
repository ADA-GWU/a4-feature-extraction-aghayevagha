import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def read_image(path):
    image_bgr = cv2.imread(path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

def resize_image(image, max_size=800):
    if image is None:
        print("Error: Could not read the image!")
        return None
    height, width = image.shape[:2]

    # If either dimension exceeds max_size
    if width > max_size or height > max_size:
        scale = min(max_size / width, max_size / height)  # pick largest dimension
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image



# Function to save the image
def save_image(image, script_dir, output_name, task_index):
    # Create the "outputs" directory inside script_dir if it doesn't exist
    outputs_folder = os.path.join(script_dir, 'outputs')
    if not os.path.exists(outputs_folder):
        os.makedirs(outputs_folder)
    
    # Create the folder "output{task_index}" inside the "outputs" folder
    output_folder = os.path.join(outputs_folder, f'output{task_index}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Define the output file path with the given output_name
    output_path = os.path.join(output_folder, output_name)

    # Save the image
    cv2.imwrite(output_path, image)
    print(f"Image saved at {output_path}")



# =======================================
                                # Part 1
# =======================================



def gaussian_filter(image, kernel_size=5, sigma=0):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
