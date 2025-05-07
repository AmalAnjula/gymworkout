
import cv2
import numpy as np
from openvino.runtime import Core
# Load the network and model
model_xml = r'open_model_zoo-master\tools\model_tools\intel\human-pose-estimation-0005\FP16\human-pose-estimation-0005.xml'
 
# Initialize inference engine
ie = Core()

# Load the model
model = ie.read_model(model=model_xml)

# Compile the model for the CPU (or other device)
compiled_model = ie.compile_model(model=model, device_name="CPU")

# Get input and output blobs
input_blob = compiled_model.input(0)
output_blob = compiled_model.output(0)

# Get input dimensions
_, _, h, w = input_blob.shape

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

# Function to extract keypoints from the model output
def extract_keypoints(output, frame_shape):
    keypoints = []
    
    # Debug: Print output shape and type
    print(f"Output shape: {output.shape}, Output type: {type(output)}")
    
    # Iterate over each detected keypoint (assuming output is [number_of_keypoints, 3])
    for keypoint in output:
        # Ensure keypoint is a numpy array or a list
        if isinstance(keypoint, (np.ndarray, list)) and len(keypoint) >= 3:
            x, y, confidence = keypoint[:3]  # Extract x, y, confidence values
            
            # Ensure confidence is a scalar (in case it's an array)
            if isinstance(confidence, np.ndarray):
                confidence = confidence.item()  # Convert array to scalar

            # Print keypoint values for debugging
            print(f"x: {x}, y: {y}, confidence: {confidence}")
 
        else:
            print(f"Unexpected keypoint format: {keypoint}")
    
    return keypoints

# Loop to process the video stream
while cap.isOpened():
    # Read frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read the frame from the webcam.")
        break

    # Resize the frame to fit the model input
    resized_frame = cv2.resize(frame, (w, h))

    # Prepare the frame for inference (Convert HWC to NCHW format)
    input_data = resized_frame.transpose((2, 0, 1))  # HWC to CHW
    input_data = input_data.reshape((1, 3, h, w))    # Add batch dimension

    # Perform inference on the frame
    result = compiled_model([input_data])[output_blob]
    
    # Debug the output shape and type
    print(f"Result shape: {result[0].shape}, Result type: {type(result[0])}")

    # Extract keypoints from the model output
    keypoints = extract_keypoints(result[0], frame.shape)

    # Draw the pose landmarks on the frame
    for (x, y) in keypoints:
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # Display the frame with pose landmarks
    cv2.imshow('Pose Estimation', frame)

    # Exit the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
