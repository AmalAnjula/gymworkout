import numpy as np

# Initialize variables
weight_initial = 70  # Initial weight in kg
calorie_deficit_per_day = -500  # Example: deficit of 500 calories per day
days = 7  # Time period for prediction

# Kalman filter parameters
weight_estimate = weight_initial
weight_estimate_error = 1  # Initial error estimate
measurement_error = 2  # Measurement noise (variance)
process_variance = 0.5  # Process variance (variance in weight change)

# Kalman filter implementation
weights = []  # To store predicted weights

for day in range(days):
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

    # Store predicted weight
    weights.append(weight_estimate)

    # Print out results
    print(f"Day {day + 1}: Predicted Weight = {predicted_weight:.2f} kg, Measured Weight = {actual_measurement:.2f} kg, Updated Weight Estimate = {weight_estimate:.2f} kg")

# Plotting the predicted weights
import matplotlib.pyplot as plt

plt.plot(range(1, days + 1), weights, marker='o', label='Estimated Weight')
plt.xlabel('Days')
plt.ylabel('Weight (kg)')
plt.title('Weight Prediction using Kalman Filter')
plt.legend()
plt.show()
