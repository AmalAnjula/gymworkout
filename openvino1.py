 
import cv2
import numpy as np
from openvino.runtime import Core

# Initialize OpenVINO Runtime and load the model
ie = Core()
model_xml = r"C:\Users\Amal Anjula\OneDrive - MMU\Subject\MSc project2\open_model_zoo-master\tools\model_tools\intel\human-pose-estimation-0001\FP16\human-pose-estimation-0001.xml"

 
model_bin = model_xml.replace(".xml", ".bin")

# Initialize OpenVINO Core and load the model
ie = Core()
model = ie.read_model(model=model_xml, weights=model_bin)
compiled_model = ie.compile_model(model, "CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Skeleton and color settings for drawing the keypoints
default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))
colors = [(255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85), (255, 0, 170), (85, 255, 0), (255, 170, 0)]

# Function to extract keypoints
def extract_keypoints(output, frame_shape, confidence_threshold=0.1):
    keypoints = []
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            x = int(output[i, 0] * frame_shape[1])
            y = int(output[i, 1] * frame_shape[0])
            keypoints.append((x, y, confidence))
    return keypoints

# Function to draw poses
def draw_poses(frame, keypoints, skeleton=default_skeleton, point_score_threshold=0.1):
    for keypoint in keypoints:
        cv2.circle(frame, keypoint[:2], 3, (0, 255, 0), -1)

    for i, j in skeleton:
        if keypoints[i][2] > point_score_threshold and keypoints[j][2] > point_score_threshold:
            cv2.line(frame, keypoints[i][:2], keypoints[j][:2], (255, 0, 0), 2)
    return frame

# Open the video capture
cap = cv2.VideoCapture(0)  # Use camera or change to a video file path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the input blob with the correct shape [1, 3, 256, 456]
    input_blob = cv2.resize(frame, (456, 256))  # Resize to 456 width, 256 height
    input_blob = input_blob.transpose((2, 0, 1))  # Transpose to [3, 256, 456]
    input_blob = np.expand_dims(input_blob, axis=0)  # Add batch dimension [1, 3, 256, 456]

    # Run inference
    result = compiled_model([input_blob])[output_layer]

    # Extract keypoints and draw them on the frame
    keypoints = extract_keypoints(result[0], frame.shape)
    frame = draw_poses(frame, keypoints)

    # Display the frame
    cv2.imshow("Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
