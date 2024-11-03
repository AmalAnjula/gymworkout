
import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import csv
import math
import time
import cv2
import joblib
import mediapipe as mp
import numpy as np
import pickle  # To load the saved model
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml  # or use xml.etree.ElementTree for XML
import warnings
 
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
clf = joblib.load('workout_classifier_model.pkl')

def readYAMLfile(main,parent):
    with open('userConfig.yaml', 'r') as file:
            data = yaml.safe_load(file)
    
    val=data[main][parent]
    #print("yaml ",main,parent,val)
    return val

# init Mediapipe Pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

 

cap=cv2.VideoCapture(readYAMLfile('workOutData','vedio_path'))
#cap = cv2.VideoCapture(r'C:\Users\Amal Anjula\OneDrive - MMU\Subject\MSc project2\trainVedio\sclkot\VID_20240915_121920.mp4')
#cap = cv2.VideoCapture(0)

workoutType=["Dumbbell latterel",'dumbbell curl','squat']


'''
The 3.5 in the formula represents the oxygen consumption at rest, measured in milliliters of oxygen per kilogram of body weight per minute (mL O₂/kg/min).

Explanation:
MET value: The metabolic equivalent of the activity you're performing (e.g., for dumbbell lateral raises, it's around 4.5).
3.5: Represents oxygen consumption in mL O₂/kg/min at rest.
Weight in kg: Your body weight in kilograms.
200: A constant used to convert oxygen consumption to calories.
Time in minutes: The duration of the workout in minutes.
Calories_burned=(MET_value*oxygen_consumption*weight*(tot_time[0]/60.00))/200.00
'''
oxygen_consumption=3.5
MET_value=[4.5,3.0,5.0]
textIns="NaN"
weight=70

# Initialize StandardScaler  
scaler = StandardScaler()

avgPick=[0,0,0,0]
lastHipPos=0
retVal=0
sholderInx=0
calRate=0
steps_count=[0,0,0,0,0]
tot_time=[0,0,0,0]
start_time = time.time()
endTime=time.time()
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

def calcul_angle(point1, point2, point3):
     
    x1=point1.x
    y1=point1.y

    x2=point2.x
    y2=point2.y

    x3=point3.x
    y3=point3.y
 
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    return angle
 

def write_calorie_burn_rate_to_file(calorie_burn_rate,takeTime):

    
    takeTime=takeTime*100.00 
    try:
        with open('userConfig.yaml', 'r') as file:
            data = yaml.safe_load(file)
            data['workOutData']['cal_burn_rate'] = calorie_burn_rate  # Change the name to the new value
            data['workOutData']['workOutDuration'] = takeTime
        
        with open('userConfig.yaml', 'w') as file:
            yaml.dump(data, file)
    except Exception as e:
        print(f"Error writing to cal.txt: {e}")

def takeDec(img,myLandmark,index):
    global lastHipPos
    global retVal,sholderInx,steps_count,start_time,endTime,calRate,tot_time
    
    Calories_burned=0
    h, w, _ = img.shape

    if index==0:


        rightUp= calcul_angle(landmarks[13], landmarks[11], landmarks[23])
        leftUp=calcul_angle(landmarks[14], landmarks[12], landmarks[24])
        
        left_elbo_coords = (int(myLandmark[13].x * w), int(myLandmark[13].y * h))
        right_elbo_coords = (int(myLandmark[14].x * w), int(myLandmark[14].y * h))


        upindx=(abs(leftUp)+abs(rightUp))/2.00
    
        offset=200

        cv2.line(img, 
                 (left_elbo_coords[0]-offset, left_elbo_coords[1] ),  # Start point (parallel)
                 (left_elbo_coords[0]+offset, left_elbo_coords[1] ),  # End point (parallel)
                 (0, 255, 0), 2)  # Color green, thickness 2
        
        cv2.line(img, 
                 (right_elbo_coords[0]-offset, right_elbo_coords[1] ),  # Start point (parallel)
                 (right_elbo_coords[0]+offset, right_elbo_coords[1] ),  # End point (parallel)
                 (0, 0, 255), 2)  # Color green, thickness 2
        

          

        if upindx > 55 and retVal==0:
            retVal=1
            start_time = time.time()
             

        elif upindx < 35 and retVal==1:
            retVal=0
            steps_count[0]=steps_count[0]+1
            endTime=time.time()-start_time      
            if endTime>0:
                tot_time[0]=tot_time[0]+endTime
                Calories_burned=(MET_value[0]*oxygen_consumption*weight*(tot_time[0]/60.00))/200.00
                calRate=Calories_burned/(tot_time[0]/60.00)
                calRate=round(calRate,1)
                write_calorie_burn_rate_to_file(calRate,tot_time[0]/3600)
                #return steps_count[0],"Time: ",round(endTime,2),"Burn rate: ",calRate,"Track: ",round(upindx,4,"tot time",round(tot_time[0],1))
                return "Step:",steps_count[0],"Step time Sec:",round(endTime,2),"Burn rate:",calRate,"tot time Sec",round(tot_time[0],1),"burned Cal:",round(Calories_burned,2)




    elif index==1:

        left_wrist_coords = (int(myLandmark[15].x * w), int(myLandmark[15].y * h))
        right_wrist_coords = (int(myLandmark[16].x * w), int(myLandmark[16].y * h))

        leftHnad=calcul_angle(landmarks[12], landmarks[14], landmarks[16])
        rightHand=calcul_angle(landmarks[11], landmarks[13], landmarks[15])
        wristInx=(leftHnad+rightHand)/2.0

        offset=200
        cv2.line(img, 
                 (left_wrist_coords[0]-offset, left_wrist_coords[1] ),  # Start point (parallel)
                 (left_wrist_coords[0]+offset, left_wrist_coords[1] ),  # End point (parallel)
                 (0, 255, 0), 2)  # Color green, thickness 2
        
        cv2.line(img, 
                 (right_wrist_coords[0]-offset, right_wrist_coords[1] ),  # Start point (parallel)
                 (right_wrist_coords[0]+offset, right_wrist_coords[1] ),  # End point (parallel)
                 (0, 0, 255), 2)  # Color green, thickness 2
        
         

        if wristInx > 55 and retVal==0:
            retVal=1
            start_time = time.time()
             

        elif wristInx < 35 and retVal==1:
            retVal=0
            steps_count[1]=steps_count[1]+1
            endTime=time.time()-start_time
        
            if endTime>0:
                tot_time[1]=tot_time[1]+endTime
                Calories_burned=(MET_value[1]*oxygen_consumption*weight*(tot_time[1]/60.00))/200.00
                calRate=Calories_burned/(tot_time[1]/60.00)
                calRate=round(calRate,1)
                write_calorie_burn_rate_to_file(calRate,tot_time[1]/3600.0)
                #return steps_count[0],"Time: ",round(endTime,2),"Burn rate: ",calRate,"Track: ",round(upindx,4,"tot time",round(tot_time[0],1))
                return "Step:",steps_count[1],"Step time Sec:",round(endTime,2),"Burn rate:",calRate,"tot time Sec",round(tot_time[1],1),"burned Cal:",round(Calories_burned,2)

        

    elif index==2:
        
        
         
        # Convert the landmark coordinates to pixel values
        left_knee_coords = (int(myLandmark[25].x * w), int(myLandmark[25].y * h))
        right_knee_coords = (int(myLandmark[26].x * w), int(myLandmark[26].y * h))

        left_wrist_coords = (int(myLandmark[15].x * w), int(myLandmark[15].y * h))
        right_wrist_coords = (int(myLandmark[16].x * w), int(myLandmark[16].y * h))

        offset=100

        cv2.line(img, 
                 (left_knee_coords[0]-offset, left_knee_coords[1] ),  # Start point (parallel)
                 (left_knee_coords[0]+offset, left_knee_coords[1] ),  # End point (parallel)
                 (0, 255, 0), 2)  # Color green, thickness 2
        
        cv2.line(img, 
                 (right_knee_coords[0]-offset, right_knee_coords[1] ),  # Start point (parallel)
                 (right_knee_coords[0]+offset, right_knee_coords[1] ),  # End point (parallel)
                 (0, 0, 255), 2)  # Color green, thickness 2
        offset=200
        cv2.line(img, 
                 (left_wrist_coords[0]-offset, left_wrist_coords[1] ),  # Start point (parallel)
                 (left_wrist_coords[0]+offset, left_wrist_coords[1] ),  # End point (parallel)
                 (0, 255, 0), 2)  # Color green, thickness 2
        
        cv2.line(img, 
                 (right_wrist_coords[0]-offset, right_wrist_coords[1] ),  # Start point (parallel)
                 (right_wrist_coords[0]+offset, right_wrist_coords[1] ),  # End point (parallel)
                 (0, 0, 255), 2)  # Color green, thickness 2

         
        sholderInx=(myLandmark[11].y+myLandmark[12].y)/2

        if sholderInx > 0.5 and retVal==0:
            retVal=1
            start_time = time.time()
             

        elif sholderInx < 0.4 and retVal==1:
            retVal=0
            steps_count[2]=steps_count[2]+1
            endTime=time.time()-start_time
           
       
            if endTime>0:
                tot_time[2]=tot_time[2]+endTime
                Calories_burned=(MET_value[2]*oxygen_consumption*weight*(tot_time[2]/60.00))/200.00
                calRate=Calories_burned/(tot_time[2]/60.00)
                calRate=round(calRate,1)
                write_calorie_burn_rate_to_file(calRate,tot_time[2]/3600.0)
                #return steps_count[0],"Time: ",round(endTime,2),"Burn rate: ",calRate,"Track: ",round(upindx,4,"tot time",round(tot_time[0],1))
                return "Step:",steps_count[2],"Step time Sec:",round(endTime,2),"Burn rate:",calRate,"tot time Sec",round(tot_time[2],1),"burned Cal:",round(Calories_burned,2)

      
        
        

         
             



    '''if index==2:
        rightLeg_=calcul_angle(myLandmark[23], myLandmark[25], myLandmark[27])
        leftLeg=calcul_angle(landmarks[24], landmarks[26], landmarks[28])
 
        kneePos=(landmarks[26].y+landmarks[25].y)/2
        if retVal==0:
            lastHipPos=kneePos
        if abs(kneePos-lastHipPos)>0.3 and retVal==0:
            retVal=1
        elif abs(kneePos-lastHipPos)>0.3 and retVal==1:
            retVal=-1
        

        return retVal'''
        
         
        
    

    
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            with open('userConfig.yaml', 'r') as file:
                data = yaml.safe_load(file)
                data['workOutData']['run'] = 0  # Change the name to the new value
            with open('userConfig.yaml', 'w') as file:
                yaml.dump(data, file)
            break
             

        # Convert the frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        # Convert back to BGR for displaying
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract pose landmarks and make prediction
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            '''rightUp = np.degrees(np.arctan2(landmarks[12].y - landmarks[14].y, landmarks[12].x - landmarks[14].x))
            rightHip = np.degrees(np.arctan2(landmarks[24].y - landmarks[26].y, landmarks[24].x - landmarks[26].x))
            rightHand = np.degrees(np.arctan2(landmarks[16].y - landmarks[14].y, landmarks[16].x - landmarks[14].x))
            rightLeg = np.degrees(np.arctan2(landmarks[26].y - landmarks[28].y, landmarks[26].x - landmarks[28].x))

            leftUp = np.degrees(np.arctan2(landmarks[11].y - landmarks[13].y, landmarks[11].x - landmarks[13].x))
            leftHip = np.degrees(np.arctan2(landmarks[23].y - landmarks[25].y, landmarks[23].x - landmarks[25].x))
            leftHand = np.degrees(np.arctan2(landmarks[15].y - landmarks[13].y, landmarks[15].x - landmarks[13].x))
            leftLeg = np.degrees(np.arctan2(landmarks[25].y - landmarks[27].y, landmarks[25].x - landmarks[27].x))'''
            
            rightUp= calcul_angle(landmarks[13], landmarks[11], landmarks[23])
            rightHip=calcul_angle(landmarks[11], landmarks[23], landmarks[25])
            rightHand=calcul_angle(landmarks[11], landmarks[13], landmarks[15])
            rightLeg=calcul_angle(landmarks[23], landmarks[25], landmarks[27])

            leftUp=calcul_angle(landmarks[14], landmarks[12], landmarks[24])
            leftHip=calcul_angle(landmarks[12], landmarks[24], landmarks[26])
            leftLeg=calcul_angle(landmarks[24], landmarks[26], landmarks[28])
            leftHnad=calcul_angle(landmarks[12], landmarks[14], landmarks[16])

            new_data = {
            'rightUp': [rightUp],
            'rightHip': [rightHip],
            'rightHand': [rightHand],
            'rightLeg': [rightLeg],
            'leftUp': [leftUp],
            'leftHip': [leftHip],
            'leftHnad': [leftHnad],
            'leftLeg': [leftLeg]
        }
            
                        # Convert the new data into a DataFrame (needed for prediction)
            new_data_df = pd.DataFrame(new_data)
            # Make the prediction
            predicted_workout = clf.predict(new_data_df)
            # Output the predicted workout type
            #print("Predicted Workout Type:", predicted_workout[0])
            prediction_confidences = clf.predict_proba(new_data_df)

            #text = f"Workout: {predicted_workout}, Confidence: {max(prediction_confidences)*100:.2f}%"
            text="nAn "
            max_confidence = np.max(prediction_confidences)
            max_confidence_index = np.argmax(prediction_confidences)
            avgPick[3]=avgPick[3]+1
            if max_confidence_index==0:
                avgPick[0]=avgPick[0]+1
            elif max_confidence_index==1:
                avgPick[1]=avgPick[1]+1
            elif max_confidence_index==2:
                avgPick[2]=avgPick[2]+1
            
            avgPick[0]=avgPick[0]/avgPick[3]
            avgPick[1]=avgPick[1]/avgPick[3]
            avgPick[2]=avgPick[2]/avgPick[3]
            

            
            text = f"raw Workout: {workoutType[max_confidence_index]}, Confidence: {max_confidence * 100:.2f}%"
            # Overlay the text on the frame
            cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            text = f"W1 : {avgPick[0] * 100:.2f}, W2: {avgPick[1] * 100 :.2f}, W3: {avgPick[2] * 100 :.2f}"
            cv2.putText(image, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            avgIndex=np.argmax(avgPick[:-1])
            text = f"Avarage prediction : {np.max(avgPick[:-1])* 100 :.2f}, index: {avgIndex}"
            
            cv2.putText(image, text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            text = f"{takeDec(image,landmarks,avgIndex)} "

            if 'None' not in text:
                textIns=text
           
            fileds=textIns.split(',')
            fileds = [field.replace("'", "").replace("(", "").replace(")", "").strip() for field in fileds]

             
            cv2.putText(image, str(fileds[:6]), (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, str(fileds[6:]), (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

             
            
            # Overlay the text on the frame
            

            #print (type(prediction_confidences))
            


            
        height, width = image.shape[:2]
        if height> 480:
            new_width = int(width * 0.5)
            new_height = int(height * 0.5)
            image = cv2.resize(image, (new_width, new_height))


        # Show the image
        if readYAMLfile('workOutData','run')== 0:
            break
            

        cv2.imshow('Workout Classification', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Exit on ESC key
            break

cap.release()
cv2.destroyAllWindows()
