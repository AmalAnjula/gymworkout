from flask import Flask, render_template_string, request, send_file
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for rendering without a display
import matplotlib.pyplot as plt
import numpy as np
import io
import yaml  # or use xml.etree.ElementTree for XML

app = Flask(__name__)

 


# Load video paths from YAML file
def load_video_paths():
    with open('userConfig.yaml', 'r') as file:
        data = yaml.safe_load(file)

    return data['videos']

# Start and stop functions
def start_function(selected_video):
    print(f"Started workout for: {selected_video['name']} ({selected_video['path']})")

def stop_function(selected_video):
    print(f"Stopped workout for: {selected_video['name']} ({selected_video['path']})")


# Function to generate weight prediction plot
def generate_weight_prediction_plot():
    weight_initial = 70  # Initial weight in kg
    calorie_burn_rate_per_hour = read_calorie_burn_rate_from_file()  # Calorie burn rate in calories per hour
    hours_per_day = 24  # 24 hours in a day
    days = 20  # Always need 7 days prediction

    # Convert calorie burn rate to daily deficit
    calorie_deficit_per_day = calorie_burn_rate_per_hour * hours_per_day

    # Kalman filter parameters
    weight_estimate = weight_initial
    weight_estimate_error = 1  # Initial error estimate
    measurement_error = 2  # Measurement noise (variance)
    process_variance = 0.5  # Process variance (variance in weight change)

    weights = []  # To store predicted weights
    time_steps = []

    # Calculate weight predictions for 7 days
    for day in range(1, days + 1):
        # Prediction step
        predicted_weight = weight_estimate - (calorie_deficit_per_day / 7700)  # Convert deficit to kg

        # Update Kalman gain
        kalman_gain = weight_estimate_error / (weight_estimate_error + measurement_error)

        # Simulate a noisy measurement (replace this with actual measurements)
        actual_measurement = np.random.normal(predicted_weight, measurement_error)
        #actual_measurement=calorie_burn_rate_per_hour
        # Update step
        weight_estimate = predicted_weight + kalman_gain * (actual_measurement - predicted_weight)

        # Update the estimate error
        weight_estimate_error = (1 - kalman_gain) * weight_estimate_error + process_variance

        # Store predicted weight and day
        weights.append(weight_estimate)
        time_steps.append(day)

    # Plot the prediction
    plt.figure(figsize=(8, 4))
    plt.plot(time_steps, weights, marker='o', label='Estimated Weight')
    plt.xlabel('Days')
    plt.ylabel('Weight (kg)')
    plt.title('7-Day Weight Prediction using Kalman Filter')
    plt.grid(True)
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
        <title>Video Workout Selection</title>
        <meta http-equiv="refresh" content="10"> <!-- Refresh every 60 seconds -->
        <style>
            body { font-family: Arial, sans-serif; margin: 50px; }
            h1 { color: #333; }
            select, button { font-size: 16px; margin: 10px; padding: 5px; }
            button { background-color: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background-color: #218838; }
            #stopBtn { background-color: #dc3545; }
            #stopBtn:hover { background-color: #c82333; }
        </style>
    </head>
    <body>
        <h1>Select Workout Video</h1>
        <form method="POST">
            <label for="video">Select a video:</label>
            <img src="/plot.png" alt="Weight Prediction Plot">
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
    ''',  videos=videos)

# Route to serve the dynamic plot image
@app.route('/plot.png')
def plot_png():
    img = generate_weight_prediction_plot()
    return send_file(img, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
