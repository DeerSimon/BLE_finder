import csv
import pandas as pd
import os
def normalize(value, min_val, max_val):
    """Normalize value to [0,1] range to ensure fair contribution."""
    return (value - min_val) / (max_val - min_val) if max_val != min_val else 0

def calculate_distance(dev, meas, min_max_values):
    """
    Compute a robust distance metric between a device and a measurement.

    Args:
    - dev: Dictionary representing the device
    - meas: Dictionary representing the measurement
    - min_max_values: Dictionary containing min/max values for normalization

    Returns:
    - Distance value
    """
    # Extract min/max values for normalization
    def min_max(feature):
        return min_max_values[feature]['min'], min_max_values[feature]['max']

    # Compute absolute differences (normalize where necessary)
    #dist_frame = abs(dev['frame_length'] - meas['frame_length'])  
    dist_frame = abs(normalize(dev['frame_length'], *min_max('frame_length')) - 
                 normalize(meas['frame_length'], *min_max('frame_length')))

    dist_signal = abs(normalize(dev['signal'], *min_max('rssi')) -
                      normalize(meas['signal'], *min_max('rssi')))              

    dist_mac_period = abs(normalize(dev['mac period'], *min_max('mac period')) - 
                          normalize(meas['mac period'], *min_max('mac period')))

    dist_occurrences = abs(normalize(dev['occurrences'], *min_max('occurrences')) - 
                           normalize(meas['occurrences'], *min_max('occurrences')))

    dist_last_seen = abs(normalize(dev['last_seen'], *min_max('last_seen')) - 
                         normalize(meas['last_seen'], *min_max('last_seen')))

    # Fingerprint similarity: 0 if same, 1 if different
    if dev['fingerprint'] == meas['fingerprint']:
        fingerprint_diff = 0
    #elif meas['fingerprint'] == 'a':
    #    fingerprint_diff = 0.2
    else:
        fingerprint_diff = 1

    # Load weights
    cureent_dir = os.path.dirname(os.path.abspath(__file__)) 
    weights_path = os.path.join(cureent_dir, 'weights.csv')
    weights_df = pd.read_csv(weights_path)

    # Convert the DataFrame to a dictionary
    weights = weights_df.to_dict(orient='records')[0]
    # Weights (adjustable for feature importance)
    #w1 = 1   #frame length
    #w2 = 1   #signal strength
    #w3 = 1   #mac period
    #w4 = 1   #occurrences
    #w5 = 1   #fingerprint
    #w6 = 1   #last seen
    

    # Compute final distance metric
    distance = (weights['frame_length'] * dist_frame 
                + 
                weights['signal'] * dist_signal 
                + 
                weights['mac period'] * dist_mac_period 
                + 
                weights['occurrences'] * dist_occurrences 
                + 
                weights['fingerprint'] * fingerprint_diff 
                #+ 
                #weights['last_seen'] * dist_last_seen
                )
    
     
    #with open('distances.csv', 'a') as f:
    #        f.write(str(distance) + '\n')
    return distance