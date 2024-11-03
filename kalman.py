from flask import Flask, render_template_string, send_file
import matplotlib.pyplot as plt
import numpy as np
import io
import threading
import time
import os

app = Flask(__name__)

# Global variable to store calorie burn rate
calorie_burn_rate_per_hour = 100  # Default value

# Function to read the calorie burn rate from the file
def read_calorie_burn_rate():
    global calorie_burn_rate_per_hour
    try:
        with open('cal.txt', 'r') as file:
            calorie_burn_rate_per_hour = float(file.read().strip())
    except Exception as e:
        print(f"Error reading cal.txt: {e}")

# Function to monitor changes in the calorie file
def monitor_calorie_file():
    last_modified_time = os.path.getmtime('cal.txt')

    while True:
        time.sleep(1)  # Check every second
        try:
            current_modified_time = os.path.getmtime('cal.txt')
            if current_modified_time != last_modified_time:
                last_modified_time = current_modified_time
                read_calorie_burn_rate()
        except Exception as e:
            print(f"Error monitoring cal.txt: {e}")

# Function to generate weight prediction plot
def generate_weight_prediction_plot():
    weight_initial = 70  # Initial weight in kg
    hours_per_day = 24  # 24 hours in a day
    days = 30  # Always need 7 days prediction

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
    plt.title(f'7-Day Weight Prediction (Calorie Burn Rate: {calorie_burn_rate_per_hour} cal/hour)')
    plt.grid(True)
    plt.legend()

    # Save plot to a BytesIO object and return it
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img

# Route to render the web page
@app.route('/')
def home():
    return render_template_string('''
    <!doctype html>
    <html>
    <head>
        <title>7-Day Weight Prediction</title>
        <meta http-equiv="refresh" content="10"> <!-- Refresh every 10 seconds -->
    </head>
    <body>
        <h1>7-Day Weight Prediction</h1>
        <img src="/plot.png" alt="Weight Prediction Plot">
    </body>
    </html>
    ''')

# Route to serve the dynamic plot image
@app.route('/plot.png')
def plot_png():
    img = generate_weight_prediction_plot()
    return send_file(img, mimetype='image/png')

if __name__ == "__main__":
    # Read initial calorie burn rate
    read_calorie_burn_rate()

    # Start a background thread to monitor changes in the calorie file
    monitor_thread = threading.Thread(target=monitor_calorie_file, daemon=True)
    monitor_thread.start()

    # Start the Flask server
    app.run(debug=True, host='0.0.0.0', port=5000)
