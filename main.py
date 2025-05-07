import subprocess
import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

from flask import Flask, render_template_string, request, send_file
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for rendering without a display
import matplotlib.pyplot as plt
import numpy as np
import io
import yaml  # or use xml.etree.ElementTree for XML

app = Flask(__name__)


def readYAMLfile(main,parent):
    with open('userConfig.yaml', 'r') as file:
            data = yaml.safe_load(file)
    
    val=data[main][parent]
    #print("yaml ",main,parent,val)
    return val

 
   # 1 kg of fat is ~7700 calories
#https://onefitness.com.au/the-real-facts-about-burning-body-fat/#:~:text=There%20are%207%2C700kcals%20(kcal,time%20to%20burn%20that%20fat.
initial_weight = readYAMLfile('personal_data','weightKG')  # kg
  
 
# Load video paths from YAML file
def load_video_paths():
    with open('userConfig.yaml', 'r') as file:
        data = yaml.safe_load(file)
    
    return data['videos']

# Start and stop functions
def start_function(selected_video):
    # Update the person's name
    with open('userConfig.yaml', 'r') as file:
        data = yaml.safe_load(file)
        data['workOutData']['run'] = 1  # Change the name to the new value
        data['workOutData']['vedio_path'] = selected_video['path']
        data['workOutData']['workOutName'] =selected_video['name']
    with open('userConfig.yaml', 'w') as file:
        yaml.dump(data, file)

    print(f"Started workout for: {selected_video['name']} ({selected_video['path']})")
    try:
        subprocess.Popen(['python3', r'C:\Users\Amal Anjula\OneDrive - MMU\Subject\MSc project2\classIdentyfy.py'])  # Replace 'another_script.py' with your script
        print("Another script started successfully.")
    except Exception as e:
        print(f"Failed to run another script: {e}")
        
def stop_function(selected_video):
    print(f"Stopped workout for: {selected_video['name']} ({selected_video['path']})")
    with open('userConfig.yaml', 'r') as file:
        data = yaml.safe_load(file)
        data['workOutData']['run'] = 0  # Change the name to the new value
    with open('userConfig.yaml', 'w') as file:
        yaml.dump(data, file)


# Function to estimate weight loss
def estimate_weight_loss(calories_burned, initial_weight):
    calories_per_kg = readYAMLfile('personal_data','calories_per_kg')

    return initial_weight - (calories_burned / calories_per_kg)
  

def generate_weight_prediction_plot():

    calories_per_min = readYAMLfile('workOutData','cal_burn_rate')  # Example calorie burn rate per minute
    total_minutes_per_day =  readYAMLfile('workOutData','workOutDuration')   # Assume 1 hour workout per day. this in minutes
    #calories_burned_per_day = calories_per_min * total_minutes_per_day
    calories_burned_per_day = calories_per_min * 60
    # Simulate daily calorie burns (for example, fluctuating over days)
    days =  readYAMLfile('personal_data','predcit_days')   

    
    daily_calories_burned = np.random.normal(loc=calories_burned_per_day, scale=50, size=days)   

    # Calculate cumulative calorie burn for each day
    cumulative_calories = np.cumsum(daily_calories_burned)

    # Initial state (weight, and assumed zero change in weight initially)
    initial_state = [initial_weight, 0]

    # Kalman Filter setup
    kf = KalmanFilter(
        transition_matrices=[[1, 1], [0, 1]],  # Simple linear system
        observation_matrices=[[1, 0]],  # We observe the weight directly
        initial_state_mean=initial_state, 
        observation_covariance=1,  # Assumed observation noise
        transition_covariance=np.eye(2) * 0.01  # Small transition noise
    )

    # Observed weight loss based on calorie burn
    observed_weights = [estimate_weight_loss(cumulative_calories[i], initial_weight) for i in range(days)]

    # Use Kalman Filter to predict future weights
    filtered_state_means, filtered_state_covariances = kf.filter(observed_weights)

    # Predict weight for next 20 days
    future_days = 20
    predicted_state_means, _ = kf.filter_update(filtered_state_means[-1], filtered_state_covariances[-1], None)

    predicted_weights = []
    for i in range(future_days):
        predicted_state_means, _ = kf.filter_update(predicted_state_means, np.eye(2) * 0.01)
        predicted_weights.append(predicted_state_means[0])

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(range(days), observed_weights, label='Observed Weight')
    plt.plot(range(days, days + future_days), predicted_weights, label='Predicted Weight', linestyle='--')
    plt.axhline(initial_weight, color='red', linestyle=':', label='Initial Weight')
    plt.xlabel('Days')
    plt.ylabel('Weight (kg)')
    plt.title('Weight Loss Prediction for 20 Days')
    plt.legend()

 # Save plot to a BytesIO object and return it
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img

 

@app.route('/', methods=['GET', 'POST'])

# Route to render the web page
def home():
    videos = load_video_paths()
     
    selected_video = None

    if request.method == 'POST':
        # Get selected video name from the form
        video_name = request.form.get('video')
        # Find the video in the list
        selected_video = next((v for v in videos if v['name'] == video_name), None)

        if 'start' in request.form and selected_video:
            start_function(selected_video)
        elif 'stop' in request.form and selected_video:
            stop_function(selected_video)

    return render_template_string('''
  <!doctype html>
<html>
<head>
    <title>Video Processing</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 50px; }
        h1 { color: #333; }
        select, button { font-size: 16px; margin: 10px; padding: 5px; }
        button { background-color: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background-color: #218838; }
        #stopBtn { background-color: #dc3545; }
        #stopBtn:hover { background-color: #c82333; }
    </style>
    
    <script>
        function reloadImage() {
            var img = document.getElementById("plotImage");
            img.src = "/plot.png?random=" + new Date().getTime();  // Add timestamp to force refresh
        }

        // Refresh the image every 1 second, for 5 seconds
        function startImageReload() {
            var interval = setInterval(reloadImage, 5000); // Refresh every 1 second

            // Stop refreshing after 5 seconds
            setTimeout(function() {
                clearInterval(interval);
            }, 5000);
        }

        // Start refreshing the image every 5 seconds (after the previous cycle ends)
        setInterval(startImageReload, 5000);
    </script>
</head>
<body>

    <form method="POST">
        <img id="plotImage" src="/plot.png" alt="Weight Prediction Plot">
        <h2>...</h2>
        <label for="video">Select a video:</label>
        
        <select name="video" id="video">
            {% for video in videos %}
                <option value="{{ video.name }}">{{ video.name }}</option>
            {% endfor %}
        </select>
        <br>
        <button type="submit" name="start">Start Workout</button>
        <button type="submit" name="stop" id="stopBtn">Stop Workout</button>
    </form>

</body>
</html>

    ''',videos=videos)

# Route to serve the dynamic plot image
@app.route('/plot.png')
def plot_png():
    img = generate_weight_prediction_plot()
    return send_file(img, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
