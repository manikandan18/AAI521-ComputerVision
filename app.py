import streamlit as st
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
import tempfile
import time  # For simulating processing time
from object_detection import detectObjects
from object_detection import detectVideo
from object_detection_count import detectObjectsAndCount
from pose_analysis import process_gif
from traffic_sign_detection import detectTrafficObjects

# Constants
MAX_FILE_SIZE_MB = 250
TABS = ["Object Detection", "Pose Analysis", "Object Counting", "Traffic Sign Detection"]

# Helper function to check file size
def check_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell() / (1024 * 1024)  # Convert to MB
    file.seek(0)  # Reset file pointer
    return file_size

# Placeholder function for processing logic
def process_file(uploaded_file, tab_name, confidence_score, progress_placeholder, class_type):
    progress_placeholder.info(f"Processing... Please wait. (Confidence Score: {confidence_score})")
    time.sleep(2)  # Simulate processing delay
    if tab_name == 'Object Detection':
        # Process Image
        if uploaded_file.name.endswith((".jpg", ".png", ".jpeg")):  # Image file
            #img = Image.open(uploaded_file)
            progress_placeholder.empty()  # Clear the "Processing..." message
            img = detectObjects(uploaded_file.name, confidence_score)
            return img, "image"
    
        # Process Video
        elif uploaded_file.name.endswith((".mp4", ".avi", ".mov", ".gif")):  # Video file
            #temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            #with open(temp_video_path, "wb") as f:
             #   f.write(uploaded_file.read())
            progress_placeholder.empty()  # Clear the "Processing..." message
            temp_video_path = detectVideo(uploaded_file.name, confidence_score)
            return temp_video_path, "video"
    
        # Unsupported file type
        else:
            progress_placeholder.empty()  # Clear the "Processing..." message
            st.error("Unsupported file format! Please upload an image or video.")
            return None, None
    elif tab_name == 'Object Counting':
         # Process Image
        if uploaded_file.name.endswith((".jpg", ".png", ".jpeg")):  # Image file
            #img = Image.open(uploaded_file)
            progress_placeholder.empty()  # Clear the "Processing..." message
            img, count = detectObjectsAndCount(uploaded_file.name, confidence_score, class_type)
            return img, "image"
    
        # Process Video
        elif uploaded_file.name.endswith((".mp4", ".avi", ".mov", ".gif")):  # Video file
            #temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            #with open(temp_video_path, "wb") as f:
             #   f.write(uploaded_file.read())
            progress_placeholder.empty()  # Clear the "Processing..." message
            temp_video_path = detectVideo(uploaded_file.name, confidence_score)
            return temp_video_path, "video"
    
        # Unsupported file type
        else:
            progress_placeholder.empty()  # Clear the "Processing..." message
            st.error("Unsupported file format! Please upload an image or video.")
            return None, None       
    elif tab_name == 'Pose Analysis':
            progress_placeholder.empty()  # Clear the "Processing..." message
            temp_video_path = process_gif(uploaded_file.name, confidence_score)
            return temp_video_path, "video"        
    elif tab_name == 'Traffic Sign Detection':
        if uploaded_file.name.endswith((".jpg", ".png", ".jpeg")):  # Image file
            #img = Image.open(uploaded_file)
            progress_placeholder.empty()  # Clear the "Processing..." message
            img = detectTrafficObjects(uploaded_file.name, confidence_score)
            return img, "image"

# Streamlit app layout
st.title("AI Video/Image Analysis Platform")
st.write("Upload an image or video and choose a tab for analysis.")

# Tabs for different functionalities
tab = st.tabs(TABS)

uploaded_file = None

for i, tab_name in enumerate(TABS):
    with tab[i]:
        st.header(tab_name)

        # File uploader
        uploaded_file = st.file_uploader(
            "Upload an Image/Video", type=["jpg", "jpeg", "png", "gif", "mp4", "avi", "mov"], key=tab_name
        )

        # Check file size
        if uploaded_file:
            file_size = check_file_size(uploaded_file)
            if file_size > MAX_FILE_SIZE_MB:
                st.error(f"File size exceeds {MAX_FILE_SIZE_MB} MB. Please upload a smaller file.")
            else:
                st.success(f"Uploaded file: {uploaded_file.name} ({file_size:.2f} MB)")

                # Confidence score input
                confidence_score = st.number_input(
                    "Adjust Confidence Score",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.01,
                    help="Set the confidence score threshold for the analysis (default: 0.5).",
                    key=f"confidence_{tab_name}",
                )

                # Additional input for "Object Counting" tab
                class_type = None
                if tab_name == "Object Counting":
                    class_type = st.text_input(
                        "Enter Class Type",
                        value="car",  # Default value, adjust as needed
                        help="Specify the class type to count (e.g., 'car', 'person').",
                        key=f"class_type_{tab_name}",
                    )

                # Process file when button is clicked
                if st.button(f"Process {tab_name}", key=f"process_{tab_name}"):
                    # Placeholder for the processing message
                    progress_placeholder = st.empty()
                    with st.spinner("Processing... Please wait."):
                        result, result_type = process_file(
                            uploaded_file,
                            tab_name,
                            confidence_score,
                            progress_placeholder,
                            class_type,  # Pass class_type to the processing function
                        )
                        if result_type == "video":
                            if result:
                                st.success(f"{tab_name} completed successfully!")
                                st.video(result)
                        if result_type == "image":
                            #if result:
                                st.success(f"{tab_name} completed successfully!")
                                st.image(result, caption=f"{tab_name} Result", use_column_width=True)
