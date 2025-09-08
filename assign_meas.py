from scipy.optimize import linear_sum_assignment
import numpy as np
from association import calculate_distance
def assign_measurements(prev_meas, tracked_devices, time, time_window):
    """
    Matches previous measurements to known devices using GNN.
    """
    if not prev_meas or not tracked_devices:
        return []
    
    num_meas = len(prev_meas)
    num_dev = len(tracked_devices)
    matrix = np.full((max(num_meas, num_dev), max(num_meas, num_dev)), 1e9)  # Initialize with large values
    prev_time = time - time_window
    for i, (meas_mac,meas) in enumerate(prev_meas.items()):
        for j, (device_mac, dev) in enumerate(tracked_devices.items()):
            if (dev['last_seen'] == time) or dev['last_seen'] == prev_time:
                continue

            if device_mac == 'e3:ba:29:71:4e:0d' and meas_mac == 'e8:76:ab:db:c4:ca':
                print('here')
            # Example of min/max values for normalization
            min_max_values = {
                'frame_length': {'min': 31, 'max': 100},
                'rssi': {'min': -100, 'max': 0},
                'mac period': {'min': 0, 'max': 10.7},  
                'occurrences': {'min': 1, 'max': 100},  
                'last_seen': {'min': 180, 'max': 360}  
            }
            #if abs(dev['frame_length'] - meas['frame_length']) <= 1:# and abs(dev['signal'] - meas['signal']) <= 35:            
            distance = calculate_distance(dev, meas, min_max_values)
            
            
            #distance = abs(dev['frame_length'] - meas['frame_length']) + abs(dev['signal'] - meas['signal'])
            
            # Apply constraints

            matrix[i, j] = distance
    
    row_ind, col_ind = linear_sum_assignment(matrix)
    
    assignments = []
    prev_meas_keys = list(prev_meas.keys())  # Extract keys from the dictionary
    
    for row, col in zip(row_ind, col_ind):
        if row < num_meas and col < num_dev and matrix[row, col] < 2:
            if row >= len(prev_meas_keys):
                print(f"        ⚠️ Row index {row} out of bounds for prev_meas_keys (len={len(prev_meas_keys)})")
                continue

            meas_key = prev_meas_keys[row]

            if meas_key not in prev_meas:
                print(f"        ⚠️ Key '{meas_key}' not found in prev_meas")
                continue

            meas = prev_meas[meas_key]
            meas['mac'] = meas_key

            tracked_keys = list(tracked_devices.keys())
            if col >= len(tracked_keys):
                print(f"        ⚠️ Column index {col} out of bounds for tracked_devices (len={len(tracked_keys)})")
                continue

            device_mac = tracked_keys[col]
            assignments.append((meas, device_mac))

    
    return assignments


