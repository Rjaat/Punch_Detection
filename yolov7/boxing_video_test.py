import torch
import cv2
import os
from utils.datasets import LoadImages
from utils.general import non_max_suppression
from utils.plots import plot_one_box

# Path to the trained model weights
weights_path = 'runs/train/boxing_detection3/weights/best.pt'

# Load the model (removing weights_only=True)
model = torch.load(weights_path)['model']
model.eval()  # Set the model to evaluation mode

# Video input (provide path to your video file)
video_path = 'Jab.webm'  # Update this to the correct relative or absolute path

# Check if the video file exists
if not os.path.exists(video_path):
    print(f"Error: {video_path} does not exist!")
    exit()

# Open the video file using OpenCV
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second (FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
output_video = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if no more frames

    # Prepare the image (resize and normalize)
    img = cv2.resize(frame, (640, 640))  # Resize frame to 640x640
    img = img[..., ::-1]  # Convert BGR to RGB
    img = torch.from_numpy(img).to('cuda')  # Move to GPU if available
    img = img.float() / 255.0  # Normalize image
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        pred = model(img)

    # Apply Non-Maximum Suppression (NMS)
    pred = pred[0]  # Get predictions
    pred = pred[pred[:, 4] > 0.5]  # Filter out low-confidence detections
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.4)[0]  # Apply NMS

    # Draw bounding boxes on the frame
    if len(pred):
        for *xyxy, conf, cls in pred:
            label = f'{names[int(cls)]} {conf:.2f}'  # Class name and confidence
            plot_one_box(xyxy, frame, label=label, color=(0, 255, 0), line_thickness=2)

    # Display the frame
    cv2.imshow('Video', frame)

    # Write the frame to the output video
    output_video.write(frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
output_video.release()
cv2.destroyAllWindows()
