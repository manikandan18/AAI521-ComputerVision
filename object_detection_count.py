from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import shutil
import cv2
import subprocess

# Method to detect object counts with given confidence score

# Helper function to get key by value from a dictionary
def getKeyByValue(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None  # Return None if value not found

# Usage in YOLO prediction
#class_name = "car"  # The class you want to filter, e.g., "car"

def detectObjectsAndCount(imageFile, confidence_score, class_type):

    # Load a pre-trained YOLOv8 model (e.g., YOLOv8n)
    model = YOLO("yolov8n.pt")  # Use other variants like yolov8s.pt for better accuracy
    
    # Define a custom directory to save the results
    custom_save_dir = "./runs/detect/predict"

    custom_read_dir = "./"
    class_index = getKeyByValue(model.names, class_type)
    count_of_class_type = 0;

    # Run inference on an image or video
    results = model.predict(
        source=f"{imageFile}",
        save=True,
        conf=confidence_score,
        save_dir=custom_save_dir,  # Specify the custom output directory
        exist_ok=True,  # Prevent creating new subdirectories
        classes=[class_index]
    )
    
    # Count the number of objects of the specified class
    for result in results:
        for box in result.boxes:
            class_name = model.names[int(box.cls)]
            if class_name == class_type:
               count_of_class_type += 1 

    # Path to the saved image in the custom directory
    image_path = f"{custom_save_dir}/{imageFile}"

    # Open the image using Pillow
    img = Image.open(image_path)

    # Prepare to draw text on the image
    draw = ImageDraw.Draw(img)
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"  # Update this path if necessary
    font_size = 30  # Specify the desired font size

    try:
       font = ImageFont.truetype(font_path, font_size)
    except OSError:
       font = ImageFont.load_default()  # You can load a custom font if needed

    # Text to display
    text = f"Number of {class_type}s are: {count_of_class_type}"

    # Define text position (at the top of the image)
    text_position = (200, 10)

    # Define text color (dark blue)
    text_color = (0, 0, 139)  # RGB value for dark blue

    # Add text to the image
    draw.text(text_position, text, fill=text_color, font=font)

    # Display the modified image with the overlay
    #plt.figure(figsize=(8, 12))
    #plt.imshow(img)
    #plt.axis("off")  # Hide axes
    #plt.show()

    return img, count_of_class_type



