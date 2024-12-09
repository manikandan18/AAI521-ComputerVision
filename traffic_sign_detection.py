from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import shutil
import cv2
import subprocess

# Method to detect objects with given confidence score

def detectTrafficObjects(imageFile, confidence_score):
    current_path = os.getcwd()
    print(f"Current working directory: {current_path}/{imageFile}")
    # Load a pre-trained YOLOv8 model (e.g., YOLOv8n)
    model = YOLO("./yolov8n_trained_traffic_signs.pt")  # Use other variants like yolov8s.pt for better accuracy
    
    # Define a custom directory to save the results
    custom_save_dir = f"{current_path}/runs/detect/predict14"

    custom_read_dir = "./"
    
    # Run inference on an image or video
    results = model.predict(
        source=f"{imageFile}",
        save=True,
        conf=confidence_score,
        save_dir=custom_save_dir,  # Specify the custom output directory
        exist_ok=True  # Prevent creating new subdirectories
    )
    
    # Display detected objects
    for result in results:
        for box in result.boxes:
            print(f"Class: {model.names[int(box.cls)]}, Confidence: {box.conf}, Coordinates: {box.xyxy}")
    output_path = f"{current_path}/output"
    #shutil.move(results[0].save_path, output_path)    
    # Path to the saved image in the custom directory
    image_path = f"{custom_save_dir}/{imageFile}"
    
    # Load and display the image
    img = mpimg.imread(image_path)
    return img
    #plt.figure(figsize=(8, 8))
    #plt.imshow(img)
    #plt.axis("off")  # Hide axes
    #plt.show()

def detectVideo(videoFile, confidence_score):
    # Load the video
    #video_path = "./Bangkok.mp4"
    cap = cv2.VideoCapture(videoFile)
    
    model = YOLO("yolov8n.pt", verbose=False)
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create a VideoWriter to save the annotated video
    output_path = f"annotated_{videoFile}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process and annotate the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        # Perform object detection
        results = model(frame)
    
        # Annotate the frame
        annotated_frame = results[0].plot()
    
        # Show the annotated frame
        #cv2.imshow("Annotated Video", annotated_frame)
    
        # Save the annotated frame to the output video
        out.write(annotated_frame)
         
        # Break on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()
    repaired_path = f"repaired_{output_path}"
    # Define the ffmpeg command
    command = [
        'ffmpeg', '-y', 
        '-i', output_path, 
        '-c:v', 'libx264', 
        '-c:a', 'aac', 
        repaired_path
    ]
    
    if retry_file_access(output_path):
      # Run the command
      try:
          subprocess.run(command, check=True)
          print("Video processed successfully")
      except subprocess.CalledProcessError as e:
          print(f"Error occurred: {e}")    
    # Release resources
    cap.release()
    out.release()
    return repaired_path
    #cv2.destroyAllWindows()
    #cv2.waitKey(1)

def retry_file_access(file_path, retries=3, delay=2):
    for i in range(retries):
        try:
            # Try accessing the file (e.g., with ffmpeg or other code)
            with open(file_path, 'rb') as file:
                # Perform the processing
                return True
        except IOError:
            # Wait before retrying
            print(f"File is not ready yet. Retrying... {i+1}/{retries}")
            time.sleep(delay)
    print("File is not accessible after multiple retries.")
    return False

