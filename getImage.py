import cv2
import os

imgcounter=0

def extract_frames_from_videos(video_dir, output_dir, trainifZero,frame_rate=30):
    global imgcounter
     
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
        
    for workout in os.listdir(video_dir):
        workout_path = os.path.join(video_dir, workout)
        if os.path.isdir(workout_path):
            workout_output_dir = os.path.join(output_dir, workout)
            if not os.path.exists(workout_output_dir):
                os.makedirs(workout_output_dir)
                
            for video_file in os.listdir(workout_path):
                video_path = os.path.join(workout_path, video_file)
                if os.path.isfile(video_path):
                    video_cap = cv2.VideoCapture(video_path)
                    count = 0
                    success = True
                    while success:
                        success, image = video_cap.read()
                        if count % frame_rate == 0 and success:
                            frame_filename = f"{os.path.splitext(video_file)[0]}_frame_{count}.jpg"
                            if trainifZero==0 and imgcounter<4:
                                cv2.imwrite(os.path.join(workout_output_dir, frame_filename), image)
                                print (f'{imgcounter} train save {frame_filename} on {workout_output_dir}')
                            elif trainifZero==1 and imgcounter==4:
                                cv2.imwrite(os.path.join(workout_output_dir, frame_filename), image)
                                print (f'{imgcounter} val save {frame_filename} on {workout_output_dir}')
                            imgcounter=imgcounter+1
                            if imgcounter==5:
                                imgcounter=0
                            
                        count += 1
                    video_cap.release()

print("--------  train ---------------------------")

video_dir = 'trainVedio'
output_dir = 'trainImages'
extract_frames_from_videos(video_dir, output_dir,0, frame_rate=30)  # Adjust frame_rate as needed train

print("--------  vali ---------------------------")

video_dir = 'trainVedio'
output_dir = 'trainImages_validation'
extract_frames_from_videos(video_dir, output_dir,1, frame_rate=30)  # Adjust frame_rate as needed valiedty
