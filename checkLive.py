import random
import threading
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load the trained model
model = tf.keras.models.load_model('workout_model.keras')

# Define the class names (update these with your actual workout class names)
class_names = ['bench', 'laterral', 'leg_press']

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()



def plot_repetition_times(repetition_times):
  # creating the figure and axes object
    fig, ax = plt.subplots()

def update_plot():

    x = [random.randint(1,100)]
    y = [random.randint(1,100)]
    x.append(random.randint(1,100))
    y.append(random.randint(1,100))
 
    ax.clear()  # clearing the axes
    ax.scatter(x,y, s = y, c = 'b', alpha = 0.5)  # creating new scatter chart with updated data
    fig.canvas.draw()  # forcing the artist to redraw itself

     



# Function to draw pose landmarks
def draw_pose_landmarks(image, landmarks, connections):
    for landmark in landmarks.landmark:
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    if connections:
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            start_point = (int(landmarks.landmark[start_idx].x * image.shape[1]), int(landmarks.landmark[start_idx].y * image.shape[0]))
            end_point = (int(landmarks.landmark[end_idx].x * image.shape[1]), int(landmarks.landmark[end_idx].y * image.shape[0]))
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)


# Function to preprocess frames
def preprocess_frame(frame, target_size=(150, 150)):
    frame = cv2.resize(frame, target_size)
    frame = frame.astype(np.float32) / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# Function to perform inference on a frame
def predict_frame(model, frame):
    preprocessed_frame = preprocess_frame(frame)
    prediction = model.predict(preprocessed_frame)
    print (prediction)
    return np.argmax(prediction), prediction


class WorkoutCounter:
    def __init__(self, image_width, image_height):
        self.reps = 0
        self.ang = 0
        self.state = 'down'  # Initial state
        self.image_width = image_width
        self.image_height = image_height
        self.time_for_rep=0
        self.start_time = time.time()
        self.repetition_times = []

    def update(self, landmarks, workout_type):
        if workout_type == class_names[1]:  # Replace with your actual workout type
            # Extract landmarks
            left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

            # Convert normalized coordinates to pixel coordinates
            elbow_x = int(left_elbow.x * self.image_width)
            elbow_y = int(left_elbow.y * self.image_height)
            shoulder_x = int(left_shoulder.x * self.image_width)
            shoulder_y = int(left_shoulder.y * self.image_height)
            wrist_x = int(left_wrist.x * self.image_width)
            wrist_y = int(left_wrist.y * self.image_height)

            # Calculate vectors
            vector_shoulder_to_elbow = np.array([elbow_x - shoulder_x, elbow_y - shoulder_y])
            vector_elbow_to_wrist = np.array([wrist_x - elbow_x, wrist_y - elbow_y])

            # Calculate angles
            angle_rad = np.arccos(np.clip(np.dot(vector_shoulder_to_elbow, vector_elbow_to_wrist) /
                                         (np.linalg.norm(vector_shoulder_to_elbow) * np.linalg.norm(vector_elbow_to_wrist)), -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)

            # Determine direction based on angle
            if angle_deg < 20 and self.state=="down" :
                self.state = 'up'
                self.reps=self.reps+1
                
                end_time = time.time()
                duration = end_time - self.start_time
                self.repetition_times.append(duration)
            elif angle_deg > 30 and self.state=="up" :
                self.state = 'down'
                self.start_time = time.time()
                

             
            #print(f"Angle: {angle_deg:.2f} degrees, State: {self.state}")


            self.ang=angle_deg

    def get_repetition_times(self):
        return self.repetition_times
    

    @staticmethod
    def calculate_angle(a, b, c):
        a = np.array([a.x, a.y])  # First point
        b = np.array([b.x, b.y])  # Mid point
        c = np.array([c.x, c.y])  # End point

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360.0 - angle
        return angle



def my_threaded_func(ar1,ar2):
    global workout_counter
    global line
    print ("start 2 thread")
    plot_repetition_times(4)
    while True:    
        time.sleep(1)
        update_plot()
        #plt.pause(0.1)  # Pause to allow for plot updates
        print ("w")

 

# Function to process video and make predictions
def process_video(video_path, model, target_width=640):
    global workout_counter
    global line
    #cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(0)  # 0 is the default camera
    frame_count = 0

    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    image_height, image_width, _ = frame_rgb.shape
    workout_counter = WorkoutCounter(image_height, image_width)
    
 


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

         # Perform pose estimation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)
        
         
         
        if result.pose_landmarks:
            draw_pose_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)


        # Perform inference on the frame
        predicted_class_idx, prediction = predict_frame(model, frame)
        predicted_class_name = class_names[predicted_class_idx]
        confidence =prediction[0][predicted_class_idx];
        confidence=round(confidence,4)
        #print (prediction[0][predicted_class_idx])
        # Display the predicted class on the frame
        
        if result.pose_landmarks:
            workout_counter.update(result.pose_landmarks, predicted_class_name)


        # Resize the frame to the target width while maintaining the aspect ratio
        aspect_ratio = frame.shape[1] / frame.shape[0]
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
        resized_frame = cv2.resize(frame, (new_width, new_height))
        confidence=round(confidence,2)
        cv2.putText(resized_frame, f'Workout: {predicted_class_name}', (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(resized_frame, f'confidence: {confidence}', (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(resized_frame, f'Reps: {workout_counter.reps } ', (10, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1, cv2.LINE_AA)
        #cv2.putText(resized_frame, f'Reps: {workout_counter.reps}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Video', resized_frame)

        


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


 

 


#video_path = r'C:\Users\Amal Anjula\OneDrive - MMU\Subject\MSc project2\trainVedio\benchPress\3191884-uhd_3840_2160_25fps.mp4'
video_path = r'C:\Users\Amal Anjula\OneDrive - MMU\Subject\MSc project2\trainVedio\lateral\3495835645-preview.mp4'
#video_path = r'C:\Users\Amal Anjula\OneDrive - MMU\Subject\MSc project2\trainVedio\lateral\1103943047-preview.mp4'

#video_path = r'C:\Users\Amal Anjula\OneDrive - MMU\Subject\MSc project2\laterrl.mp4'

# Process the video and display predictions in real-time
process_video(video_path, model)
