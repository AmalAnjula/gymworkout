import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import matplotlib.pyplot as plt
import time
import threading

# Load the trained model
model = tf.keras.models.load_model('workout_model.keras')

# Define the class names (update these with your actual workout class names)
class_names = ['bench', 'laterral', 'leg_press']

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# Kalman filter implementation
weights = []  # To store predicted weights
waitPredicGraph=1

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
    #print(prediction)
    return np.argmax(prediction), prediction


class WorkoutCounter:
    def __init__(self, image_width, image_height):
        self.reps = 0
        self.state = 'down'  # Initial state
        self.image_width = image_width
        self.image_height = image_height
        self.start_time = time.time()
        self.repetition_times = []

    def update(self, landmarks, workout_type):
        if workout_type == class_names[1]:  # Replace with your actual workout type
            # Extract landmarks
            left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

            # Convert normalized coordinates to pixel coordinates
            elbow_x = int(left_elbow.x * self.image_width)
            elbow_y = int(left_elbow.y * self.image_height)
            shoulder_x = int(left_shoulder.x * self.image_width)
            shoulder_y = int(left_shoulder.y * self.image_height)

            # Calculate vectors
            vector_shoulder_to_elbow = np.array([elbow_x - shoulder_x, elbow_y - shoulder_y])
            # Calculate angle
            angle_rad = np.arccos(np.clip(vector_shoulder_to_elbow[1] / np.linalg.norm(vector_shoulder_to_elbow), -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)

            # Determine direction based on angle
            if angle_deg < 20 and self.state == "down":
                self.state = 'up'
                self.reps += 1
                end_time = time.time()
                duration = end_time - self.start_time
                self.repetition_times.append(duration)
            elif angle_deg > 30 and self.state == "up":
                self.state = 'down'
                self.start_time = time.time()

    def get_repetition_times(self):
        return self.repetition_times


def plot_weight_prediction(stop_event):
    global waitPredicGraph
    weight_initial = 70  # Initial weight in kg
    calorie_deficit_per_day = -500  # Example: deficit of 500 calories per day
    days = 30  # Total days for prediction

    # Kalman filter parameters
    weight_estimate = weight_initial
    weight_estimate_error = 1  # Initial error estimate
    measurement_error = 2  # Measurement noise (variance)
    process_variance = 0.5  # Process variance (variance in weight change)

    # Initialize plot for weight prediction
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('Days')
    ax.set_ylabel('Weight (kg)')
    ax.set_title('Live Weight Prediction using Kalman Filter')

    weights = []  # To store predicted weights
    time_steps = []

    day = 0
# Plot update
    ax.clear()
    ax.set_xlabel('Days')
    ax.set_ylabel('Weight (kg)')
    ax.set_title('Live Weight Prediction using Kalman Filter')
    plt.legend()
    while day < days and not stop_event.is_set():
        # Prediction step
        predicted_weight = weight_estimate + (calorie_deficit_per_day / 7700)  # Convert deficit to kg

        # Update Kalman gain
        kalman_gain = weight_estimate_error / (weight_estimate_error + measurement_error)

        # Simulate a noisy measurement (replace this with actual measurements)
        actual_measurement = np.random.normal(predicted_weight, measurement_error)

        # Update step
        weight_estimate = predicted_weight + kalman_gain * (actual_measurement - predicted_weight)

        # Update the estimate error
        weight_estimate_error = (1 - kalman_gain) * weight_estimate_error + process_variance

        # Store predicted weight and day
        weights.append(weight_estimate)
        time_steps.append(day + 1)

        
        ax.plot(time_steps, weights, marker='o', label='Estimated Weight')
        
        #plt.pause(0.1)

        day += 1
        #time.sleep(0.1)  # Sleep to simulate time delay

        waitPredicGraph=1
        while(waitPredicGraph):
            pass

    plt.ioff()
    plt.wait()

    


def plot_repetition_times(workout_counter, stop_event):
    global waitPredicGraph
    lastTime=0
    plt.ion()  # Turn interactive mode on
    fig, ax = plt.subplots(figsize=(4, 2))  # Example: 8 inches wide, 4 inches tall
   
    
    line, = ax.plot([], [], 'b-o')
    ax.set_title('Repetition Duration Over Time')
    ax.set_xlabel('Repetition Number')
    ax.set_ylabel('Duration (seconds)')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.grid(True)

    while not stop_event.is_set():
        times = workout_counter.get_repetition_times()
        line.set_xdata(np.arange(len(times)))
        line.set_ydata(times)
        ax.set_xlim(0, len(times) + 1)
        ax.set_ylim(0, max(times) if times else 1)
        plt.pause(0.1)  # Pause to allow for plot updates
        if lastTime!=times:
            lastTime=times
            waitPredicGraph=0

    plt.close(fig)  # Close the plot when stopping

def process_video(video_path, model, target_width=640):
    cap = cv2.VideoCapture(video_path)
    #cap = cv2.VideoCapture(0)
    frame_count = 0

    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    image_height, image_width, _ = frame_rgb.shape
    workout_counter = WorkoutCounter(image_width, image_height)

    # Create a threading event to stop the plot thread
    stop_event = threading.Event()

    # Start the plotting thread
    plot_thread = threading.Thread(target=plot_repetition_times, args=(workout_counter, stop_event))
    plot_thread.start()

    plot_thread = threading.Thread(target=plot_weight_prediction,  args=(stop_event,))
    plot_thread.start()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Perform pose estimation
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)
            
            if result.pose_landmarks:
                # Draw landmarks and connections on the frame
                for landmark in result.pose_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    start_point = (int(result.pose_landmarks.landmark[start_idx].x * frame.shape[1]),
                                   int(result.pose_landmarks.landmark[start_idx].y * frame.shape[0]))
                    end_point = (int(result.pose_landmarks.landmark[end_idx].x * frame.shape[1]),
                                 int(result.pose_landmarks.landmark[end_idx].y * frame.shape[0]))
                    cv2.line(frame, start_point, end_point, (255, 255, 255), 3)

            # Perform inference on the frame
            predicted_class_idx, prediction = predict_frame(model, frame)
            predicted_class_name = class_names[predicted_class_idx]
            confidence = prediction[0][predicted_class_idx]
            confidence = round(confidence, 4)

            if result.pose_landmarks:
                workout_counter.update(result.pose_landmarks, predicted_class_name)

            # Resize the frame to the target width while maintaining the aspect ratio
            aspect_ratio = frame.shape[1] / frame.shape[0]
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
            resized_frame = cv2.resize(frame, (new_width, new_height))
            resized_frame=frame
            cv2.putText(resized_frame, f'Workout: {predicted_class_name}', (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(resized_frame, f'confidence: {confidence}', (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(resized_frame, f'Reps: {workout_counter.reps}', (10, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1, cv2.LINE_AA)

            # Display the frame
            cv2.imshow('Video', resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # When done, signal the plot thread to stop
        print("all sroped -------------------")
        stop_event.set()
        plot_thread.join()

    cap.release()
    cv2.destroyAllWindows()

# Example usage
#video_path = r'C:\Users\Amal Anjula\OneDrive - MMU\Subject\MSc project2\trainVedio\lateral\3495835645-preview.mp4'
#video_path = r'C:\Users\Amal Anjula\OneDrive - MMU\Subject\MSc project2\trainVedio\benchPress\3191884-uhd_3840_2160_25fps.mp4'
#video_path = r'C:\Users\Amal Anjula\OneDrive - MMU\Subject\MSc project2\trainVedio\lateral\3495835645-preview.mp4'
#video_path = r'C:\Users\Amal Anjula\OneDrive - MMU\Subject\MSc project2\trainVedio\lateral\1103943047-preview.mp4'

video_path = r'C:\Users\Amal Anjula\OneDrive - MMU\Subject\MSc project2\laterrl.mp4'

process_video(video_path, model)
