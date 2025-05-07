import cv2
import mediapipe as mp
import numpy as np
import csv

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, mark the image as not writeable to pass by reference
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Right side landmarks
            right_wrist = [landmarks[16].x, landmarks[16].y]
            right_elbow = [landmarks[14].x, landmarks[14].y]
            right_shoulder = [landmarks[12].x, landmarks[12].y]
            
            right_ankle = [landmarks[28].x, landmarks[28].y]
            right_knee = [landmarks[26].x, landmarks[26].y]
            right_hip = [landmarks[24].x, landmarks[24].y]
            
            # Left side landmarks
            left_wrist = [landmarks[15].x, landmarks[15].y]
            left_elbow = [landmarks[13].x, landmarks[13].y]
            left_shoulder = [landmarks[11].x, landmarks[11].y]

           
            left_ankle = [landmarks[27].x, landmarks[27].y]
            left_knee = [landmarks[25].x, landmarks[25].y]
            left_hip = [landmarks[23].x, landmarks[23].y]
            

            # Print out coordinates
            '''print(f"Right wrist: {right_wrist}, Right elbow: {right_elbow}, Right shoulder: {right_shoulder}")
            print(f"Right hip: {right_hip}, Right knee: {right_knee}, Right ankle: {right_ankle}")
            print(f"Left wrist: {left_wrist}, Left elbow: {left_elbow}, Left shoulder: {left_shoulder}")
            print(f"Left hip: {left_hip}, Left knee: {left_knee}, Left ankle: {left_ankle}")'''

            # Draw the landmarks and connections
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv2.putText(image, f'rightDown: {(rightDown)}',  (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'rightDown: {(rightDown)}',  (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'leftUp: {(leftUp)}',  (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'leftDown: {(leftDown)}',  (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
        # Flip the image horizontally for a selfie-view display.
        #cv2.imshow('MediaPipe Pose', cv2.flip(image,0))
        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
