import math
import os
import time
import cv2
import mediapipe as mp
import numpy as np
import csv

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
csv_file = 'joint_angles.csv'
saveNow=0
frame=0


start_time = time.time()


def writeData(rightUp,rightHip,rightHand,rightLeg,leftUp,leftHip,leftHnad,leftLeg):
    global frame
    frame=frame+1
    #  writeData(rightUp,rightHip,rightHand,rightLeg,leftUp,leftHip,leftHnad,leftLeg)
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        csv_writer = csv.writer(f)
        if not file_exists:
            csv_writer.writerow(["rightUp","rightHip","rightHand","rightLeg","leftUp","leftHip","leftHnad","leftLeg",
                                  "frame","workout"
                                  ])

        csv_writer.writerow([rightUp,rightHip,rightHand,rightLeg,leftUp,leftHip,leftHnad,leftLeg,"-1","2"])


def calcul_angle(point1, point2, point3):
     
    x1=point1.x
    y1=point1.y

    x2=point2.x
    y2=point2.y

    x3=point3.x
    y3=point3.y
 
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    return angle

    
 

# For webcam input:
cap = cv2.VideoCapture(r'trainVedio/lat front/VID_20240915_121358.mp4')

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
            '''right_wrist = [landmarks[16].x, landmarks[16].y]
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
            left_hip = [landmarks[23].x, landmarks[23].y]'''

            rightUp= calcul_angle(landmarks[13], landmarks[11], landmarks[23])
            rightHip=calcul_angle(landmarks[11], landmarks[23], landmarks[25])
            rightHand=calcul_angle(landmarks[11], landmarks[13], landmarks[15])
            rightLeg=calcul_angle(landmarks[23], landmarks[25], landmarks[27])

            leftUp=calcul_angle(landmarks[14], landmarks[12], landmarks[24])
            leftHip=calcul_angle(landmarks[12], landmarks[24], landmarks[26])
            leftHnad=calcul_angle(landmarks[12], landmarks[14], landmarks[16])
            leftLeg=calcul_angle(landmarks[24], landmarks[26], landmarks[28])






            if(saveNow):
                writeData(rightUp,rightHip,rightHand,rightLeg,leftUp,leftHip,leftHnad,leftLeg)
            
            elapsed_time = time.time() - start_time
            
            cv2.putText(image, f'saving: {(saveNow)} time {elapsed_time:.2f}',  (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
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

            height, width = image.shape[:2]
            height=height*landmarks[13].y
            width=width*landmarks[13].x
             
            cv2.putText(image, str(round(leftHnad,3)),  (int(width ), int(height)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255, 0), 5, cv2.LINE_AA)
            
  


        '''cv2.putText(image, f'rightDown: {(rightDown)}',  (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'rightDown: {(rightDown)}',  (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'leftUp: {(leftUp)}',  (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'leftDown: {(leftDown)}',  (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)'''
        
        
        # Flip the image horizontally for a selfie-view display.
        #cv2.imshow('MediaPipe Pose', cv2.flip(image,0))


        
        height, width = image.shape[:2]
        new_width = int(width * 0.5)
        new_height = int(height * 0.5)
        resized_frame = cv2.resize(image, (new_width, new_height))


        cv2.imshow('MediaPipe Pose', resized_frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # 's' key
            print("Save frame")
            saveNow=1
        elif key == ord('d'):  # 's' key
            print("stop frame")
            saveNow=0


cap.release()
cv2.destroyAllWindows()
