import matplotlib.pyplot as plt
import numpy as np

# RSSI to Distance Conversion Function
#def rssi_to_distance(rssi, rssi_0=-40, n=2.7):
#    return 10 ** ((rssi_0 - rssi) / (10 * n))

import numpy as np

def rssi_to_distance(rssi, transmit_power=0, antenna_gain=0, n=2.7):
    # Receiver sensitivity LE 1M is already factored into RSSI value.
    # We are assuming that the transmit power is given in dBm and antenna gain is in dBi.
    # 'mp' is the received power at 1 meter distance.
    mp = transmit_power + antenna_gain + antenna_gain - 40 # -40 is an example value, calibrate it for your specific device
    distance = 10 ** ((mp - rssi) / (10 * n))
    return distance

if __name__ == "__main__":
    # Generate RSSI values from -100 dBm to -20 dBm
    rssi_values = np.linspace(-105, -20, 500)
    distance_values = rssi_to_distance(rssi_values)

    # Plot the curve
    plt.figure(figsize=(8, 5))
    plt.plot(rssi_values, distance_values, label="RSSI to Distance Curve", color='blue')
    plt.xlabel("RSSI (dBm)")
    plt.ylabel("Estimated Distance (meters)")
    plt.title("RSSI to Distance Conversion")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()
