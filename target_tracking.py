import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mac_path import find_mac_path
from attributes import group_data, process_grouped_data, most_frequent_frame_length_per_mac
from proccess_meas import process_measurements
from matplotlib.ticker import ScalarFormatter
import os
import json


def get_metadata_path():
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, 'metadata.json')


def load_metadata():
    path = get_metadata_path()
    if os.path.exists(path):
        with open(path, 'r') as f:
            meta = json.load(f)
            # Restore grouped as DataFrame if available
            grouped = pd.DataFrame(meta['grouped']) if 'grouped' in meta else pd.DataFrame()
            # Load persistent classifications indexed by (mac, time_window) tuple
            persistent_classifications = meta.get('persistent_classifications', {})
            # Convert string keys back to tuples
            persistent_classifications = {eval(k): v for k, v in persistent_classifications.items()}
            # Load device updates
            device_updates = meta.get('device_updates', [])
            return {
                'tracked_devices': meta.get('tracked_devices', {}),
                'prev_meas': meta.get('prev_meas', {}),
                'prev_prev_meas': meta.get('prev_prev_meas', {}),
                'device_counter': meta.get('device_counter', 1),
                'last_index_processed': meta.get('last_index_processed', -1),
                'grouped': grouped,
                'persistent_classifications': persistent_classifications,
                'device_updates': device_updates
            }
    return {
        'tracked_devices': {},
        'prev_meas': {},
        'prev_prev_meas': {},
        'device_counter': 1,
        'last_index_processed': -1,
        'grouped': pd.DataFrame(),
        'persistent_classifications': {},
        'device_updates': []
    }


def save_metadata(tracked_devices, prev_meas, prev_prev_meas, device_counter, 
                 last_index_processed, grouped, persistent_classifications, device_updates):
    path = get_metadata_path()
    tracked_devices = {str(k): v for k, v in tracked_devices.items()}
    prev_meas = {str(k): v for k, v in prev_meas.items()}
    prev_prev_meas = {str(k): v for k, v in prev_prev_meas.items()}
    grouped_dict = grouped.to_dict(orient='list')
    # Convert tuple keys to strings for JSON compatibility
    persistent_classifications_str = {str(k): v for k, v in persistent_classifications.items()}
    
    metadata = {
        'tracked_devices': tracked_devices,
        'prev_meas': prev_meas,
        'prev_prev_meas': prev_prev_meas,
        'device_counter': device_counter,
        'last_index_processed': last_index_processed,
        'grouped': grouped_dict,
        'persistent_classifications': persistent_classifications_str,
        'device_updates': device_updates
    }
    with open(path, 'w') as f:
        json.dump(metadata, f)


def update_retroactive_classifications(persistent_classifications, mac_updates, old_classification, new_classification):
    """Update classifications retroactively when a device ID changes"""
    for key, classification in persistent_classifications.items():
        if classification == old_classification:
            persistent_classifications[key] = new_classification


def target_tracking(data):
    metadata = load_metadata()
    tracked_devices = metadata['tracked_devices']
    prev_meas = metadata['prev_meas']
    prev_prev_meas = metadata['prev_prev_meas']
    device_counter = metadata['device_counter']
    last_index_processed = metadata['last_index_processed']
    previous_grouped = metadata['grouped']
    persistent_classifications = metadata['persistent_classifications']
    device_updates = metadata['device_updates']  # Load existing device updates

    # Clean the data
    if 'Info' in data.columns and data['Info'].notna().any():
        data = data[~data['Info'].str.contains('SCAN_REQ', na=False, case=False)]
        data = data[~data['Info'].str.contains('SCAN_RSP', na=False, case=False)]
        data = data[~data['Info'].str.contains('ADV_IND[Malformed Packet]', na=False, case=False)]
    if data['RSSI'].dtype == 'object' and data['RSSI'].str.endswith(' dBm').any():
        data['RSSI'] = pd.to_numeric(data['RSSI'].str.replace(' dBm', '', regex=False))

    # Process all data - no time filtering
    if data.empty:
        print("No data to process.")
        return previous_grouped, tracked_devices

    time_slice = 5

    # Apply most_frequent_frame_length_per_mac to all data
    data = most_frequent_frame_length_per_mac(data)
    
    # Initialize tracking for first run only
    if not tracked_devices:
        filtered_data = data[data['Time'] <= (min(data['Time']) + time_slice)]
        grouped_targets = process_grouped_data(filtered_data, time_slice)

        tracked_devices = {
            meas['Mac']: {
                'id': idx,
                'signal': meas['RSSI'],
                'frame_length': meas['Frame_length'],
                'mac period': meas['MAC Period'],
                'fingerprint': meas['Fingerprint'],
                'occurrences': meas['Occurrences'],
                'last_seen': 0
            }
            for idx, meas in grouped_targets.iterrows()
        }
        device_counter = len(tracked_devices) + 1

    time_window = time_slice * 6
    drop_timeout = time_window * 10000000
    treshold_occurrences = (3 / 60)

    data['Time Window'] = np.floor(data['Time'] / time_window) * time_window
    data = data.sort_values(by='Time')

    # Group all the data
    grouped = group_data(data, time_window, 'Time Window')
    grouped = grouped[grouped['Occurrences'] > treshold_occurrences].reset_index(drop=True)

    # Create results array for ALL grouped data
    results = [None] * len(grouped)
    
    # Restore previous classifications for ALL rows that match previous data
    if not previous_grouped.empty:
        for i in range(len(grouped)):
            current_row = grouped.iloc[i]
            # Look for exact match in previous_grouped by mac and time window
            matching_rows = previous_grouped[
                (previous_grouped['Mac'] == current_row['Mac']) & 
                (previous_grouped['Time Window'] == current_row['Time Window'])
            ]
            if not matching_rows.empty:
                results[i] = matching_rows.iloc[0]['Classification']

    # Set up processing parameters
    total = len(grouped)
    
    # Determine where to start actual processing (after last processed index)
    processing_start_index = last_index_processed + 1

    # Process ALL rows in grouped, but only do computation for new rows
    for i, meas in grouped.iterrows():
        
        # Skip already processed rows - but keep them available for retroactive updates
        if i < processing_start_index:
            continue
        
        print(f"Processing {i+1}/{total} ({(i+1)/total:.1%} complete)")

        mac = meas['Mac']
        signal = meas['RSSI']
        frame_length = meas['Frame_length']
        time = meas['Time Window']
        time_gap = meas['MAC Period']
        occurrences = meas['Occurrences']
        fingerprint = meas['Fingerprint']

        if mac == '0' or mac == '4d:ff:75:5f:78:7a' or time in [420, 600, 6240]:
            print(mac, time)

        # Process measurements that are ready
        to_process = {}
        to_delete = []

        for prev_mac, prev_meas in prev_prev_meas.items():
            if prev_meas['last_seen'] <= time - 2 * time_window:
                to_process[prev_mac] = prev_meas
                to_delete.append(prev_mac)

        if to_process:
            # Create a temporary results array that's large enough for old indices
            max_old_idx = max([meas['idx'] for meas in to_process.values()], default=-1)
            temp_results = [None] * max(len(results), max_old_idx + 1)
            
            # Process delayed measurements
            tracked_devices, device_counter = process_measurements(
                time, to_process, tracked_devices, device_updates, temp_results, device_counter, time_window
            )
            
            # Update persistent classifications from temp_results
            for mac_key, meas_data in to_process.items():
                if temp_results[meas_data['idx']] is not None:
                    persistent_classifications[(mac_key, meas_data['last_seen'])] = temp_results[meas_data['idx']]

        # Clean up processed measurements
        for key in to_delete:
            del prev_prev_meas[key]

        # Remove old tracked devices
        tracked_devices = {
            mac_addr: details for mac_addr, details in tracked_devices.items()
            if time - details['last_seen'] <= drop_timeout
        }

        # Check for device updates that affect current MAC
        matching_updates = [dev_update for dev_update in device_updates if dev_update['old'] == mac]

        if mac in tracked_devices:
            # Update existing tracked device
            tracked_devices[mac]['signal'] = signal
            tracked_devices[mac]['frame_length'] = frame_length
            tracked_devices[mac]['last_seen'] = time
            tracked_devices[mac]['mac period'] = time_gap
            tracked_devices[mac]['occurrences'] = occurrences
            tracked_devices[mac]['fingerprint'] = fingerprint
            results[i] = tracked_devices[mac]['id']
            persistent_classifications[(mac, time)] = tracked_devices[mac]['id']

        elif matching_updates:
            print(f"{mac}, idx: {i} exists in device_updates['old']: {matching_updates}")
            id_path = find_mac_path(matching_updates[0]['new'], device_updates)
            
            # Find ALL indices in current grouped data that need retroactive updating
            indices = grouped.loc[
                (grouped['Mac'].isin(id_path)) &
                (grouped.index >= matching_updates[0]['idx']) &
                (grouped.index < i)
            ].index.tolist()
            
            # Store the old device ID before making changes
            old_device_id = tracked_devices[id_path[-1]]['id']
            
            # Retroactively update classifications for the wrongly classified path with new device counter
            for y in indices:
                if y < len(results):  # Safety check
                    results[y] = device_counter
                    # Update persistent classifications
                    mac_key = grouped.loc[y, 'Mac']
                    time_key = grouped.loc[y, 'Time Window']
                    persistent_classifications[(mac_key, time_key)] = device_counter

            # Create new tracked device for current MAC with the old (correct) ID
            tracked_devices[matching_updates[0]['old']] = {
                'id': old_device_id,
                'signal': signal,
                'frame_length': frame_length,
                'mac period': time_gap,
                'fingerprint': fingerprint,
                'occurrences': occurrences,
                'last_seen': time
            }
            
            # Assign new device ID to the wrongly classified target
            tracked_devices[id_path[-1]]['id'] = device_counter
            
            # Update all other persistent classifications that had the old wrong ID
            update_retroactive_classifications(persistent_classifications, device_updates, old_device_id, device_counter)
            
            device_counter += 1
            results[i] = tracked_devices[matching_updates[0]['old']]['id']
            persistent_classifications[(mac, time)] = tracked_devices[matching_updates[0]['old']]['id']
            matching_updates = []

        elif mac not in prev_prev_meas:
            # Add to delayed processing queue
            prev_prev_meas[mac] = {
                'signal': signal,
                'frame_length': frame_length,
                'last_seen': time,
                'mac period': time_gap,
                'occurrences': occurrences,
                'fingerprint': fingerprint,
                'idx': i  # Current index for process_measurements compatibility
            }

    # Assign all results to grouped DataFrame
    grouped['Classification'] = results

    # Update the last processed index to the last row we processed
    new_last_index_processed = len(grouped) - 1

    # Save metadata with updated state including device_updates
    save_metadata(tracked_devices, prev_meas, prev_prev_meas, device_counter, 
                 new_last_index_processed, grouped, persistent_classifications, device_updates)

    return grouped, tracked_devices

    #grouped.to_csv('grouped_data.csv', index=False)


def plot_data(grouped):
    grouped = grouped.dropna(subset=['Classification'])
    
    # Filter for specific classifications
    selected_classifications = [29,84, 118] #[0, 18, 30, 16, 29, 34, 13, 12, 9, 25, 37, 32, 38, 27, 24, 14, 40, 1, 26, 19, 6, 15, 35, 3, 43, 41, 33, 36, 21, 47, 46, 2, 61, 65, 67, 68, 64, 20, 28, 114, 127, 131, 130, 128, 133, 139, 134, 145, 138, 144, 119, 42, 182]
    grouped = grouped[grouped['Classification'].isin(selected_classifications)]
    
    # Check if we have any data left after filtering
    if grouped.empty:
        print("No data available for the selected classifications.")
        return
    
    grouped = grouped.sort_values(['Time Window'])
    
    # Visualization
    n_classes = len(grouped['Classification'].unique())  # Number of unique classifications
    colors = plt.cm.get_cmap('hsv', n_classes)  # Use 'hsv' colormap for vibrant distinct colors
    colors = plt.cm.get_cmap('jet', n_classes)  # Use 'jet' colormap for more distinct colors

    plt.figure(figsize=(14, 8))

    # Scatter plot with colormap
    scatter = plt.scatter(
        grouped['Time Window'], 
        grouped['Mac'],  
        alpha=0.7, 
        c=grouped['Classification'], 
        cmap=colors
    )

    # Get colormap and normalize it
    norm = plt.Normalize(vmin=grouped['Classification'].min(), vmax=grouped['Classification'].max())
    cmap = cm.get_cmap(colors)

    # Connect points within the same class with correct colors
    for class_label in grouped['Classification'].unique():
        subset = grouped[grouped['Classification'] == class_label]
        plt.plot(
            subset['Time Window'], 
            subset['Mac'], 
            marker='o', 
            linestyle='-', 
            alpha=0.5, 
            color=cmap(norm(class_label))  # Use colormap to get correct color
        )

    # Add color bar
    plt.colorbar(scatter, label='Classification', ticks=np.arange(1, len(grouped['Classification'].unique())+1))
    
    # Add legend and title
    #plt.title('Data Points by Classification')
    plt.xlabel('Time [s]')
    plt.ylabel('Source')
    
    plt.show()

# Usage:
# plot_data(grouped)
    #grouped = grouped.sort_values(by='Classification')

#    plt.figure(figsize=(14, 8))
#
#    # Plot all points with the large colormap
#    scatter = plt.scatter(
#        grouped['Classification'], 
#        grouped['Mac'],  
#        alpha=0.7, 
#        c=grouped['RSSI'], 
#        cmap=colors
#    )
#
#
#    # Add color bar to represent classifications
#    plt.colorbar(scatter, label='RSSI')
#
#    # Add legend and title
#    plt.title('Data Points by Classification')
#    plt.xlabel('Device ID')
#    plt.ylabel('Source')
#    plt.show()
#
#
#
#    colors = plt.cm.get_cmap('jet', n_classes)
#    plt.figure(figsize=(14, 8))
#
#    # Plot all points with the large colormap
#    scatter = plt.scatter(
#        grouped['Classification'], 
#        grouped['Occurrences'],  
#        alpha=0.7, 
#        c=grouped['MAC Period'], 
#        cmap=colors
#    )
#
#
#    # Add color bar to represent classifications
#    plt.colorbar(scatter, label='GAPS BETWEEN MEASUREMENTS')
#
#    # Add legend and title
#    plt.title('Data Points by Classification')
#    plt.xlabel('Device ID')
#    plt.ylabel('Frequency of Occurrence') 
#    plt.show()
#
#    #------------------------------------------------------------------#
#    # Generate 35 random colors
#    random_colors = np.random.rand(n_classes, 3)  # RGB values between 0 and 1
#
#    # Create a ListedColormap
#    colors = mcolors.ListedColormap(random_colors)
#
#    plt.figure(figsize=(14, 8))
#
#    # Scatter plot with random colors
#    scatter = plt.scatter(
#        grouped['Classification'], 
#        grouped['Mac'],  
#        alpha=0.7, 
#        c=grouped['Classification'], 
#        cmap=colors
#    )
#    # Get colormap and normalize it
#    norm = plt.Normalize(vmin=grouped['Classification'].min(), vmax=grouped['Classification'].max())
#    cmap = cm.get_cmap(colors)
#    # Connect points within the same class
#    for class_label in grouped['Classification'].unique():
#        subset = grouped[grouped['Classification'] == class_label]
#        plt.plot(subset['Classification'], subset['Mac'], marker='o', linestyle='-', alpha=0.5,color=cmap(norm(class_label)))  # Use colormap to get correct color)
#
#    # Add color bar
#    plt.colorbar(scatter, label='Classification')
#
#    # Add legend and title
#    plt.title('Data Points by Classification')
#    plt.xlabel('Device ID')
#    plt.ylabel('Frame Length')
#    plt.show()
#
#
#    
#
#
#    plt.figure(figsize=(10, 6))
#    for classification in grouped['Classification'].unique():
#        filtered_data = grouped[grouped['Classification'] == classification]
#        plt.plot(filtered_data['Time Window'], filtered_data['MAC Period'], label=classification)
#
#    plt.title("MAC Period over Time for all Classifications")
#    plt.xlabel("Time Window")
#    plt.ylabel("MAC Period")
#
#    # Get current axis
#    ax = plt.gca()
#
#    # Force plain formatting for y-axis
#    formatter = ScalarFormatter(useOffset=False, useMathText=False)
#    formatter.set_scientific(False)
#    ax.yaxis.set_major_formatter(formatter)
#
#    plt.legend()
#    plt.tight_layout()
#    plt.show()

if __name__ == "__main__":
    #filepath = "D:\\LIU\\Diploma\\diplom\\code\\overcooked_37_38_39.csv" #home_meas_37_vol1 train_6_C_Kitchen overcooked_37_38_39
    #filepath="C:\\Users\\jelin\\Desktop\\zk\\iphone_11_ground.csv"
    #filepath="C:\\Users\\jelin\\Desktop\\zk\\iphone_12_pro_max_chinese.csv"
    #filepath="C:\\Users\\jelin\\Desktop\\zk\\Xiomi_12_PRO_kostas_connected_to_speaker.csv"
    #filepath="C:\\Users\\jelin\\Desktop\\zk\\radans_phone_scan_req_only.csv"
    #filepath="C:\\Users\\jelin\\Desktop\\zk\\Simulation_1_1a.csv"
    #filepath="C:\\Users\\jelin\\Desktop\\zk\\quite_1_1.csv"
    #filepath="C:\\Users\\jelin\\Desktop\\zk\\sim_all.csv"
    #filepath="C:\\Users\\jelin\\Desktop\\zk\\iphone_12_mini_ground.csv"
    #filepath = "C:\\Users\\jelin\\Desktop\\zk\\iphone_11_ground_sep.csv"
    #filepath = "C:\\Users\\jelin\\Desktop\\zk\\iphone_11_ground_hop.csv"
    #filepath="position_est_100.csv"
    
    #test
    #filepath = "C:\\Users\\jelin\\Desktop\\zk\\test\\iphone_12_mini_ground.csv"
    #filepath = "C:\\Users\\jelin\\Desktop\\zk\\test\\iphone_12_mini_lubi_ground.csv"
    #filepath = "C:\\Users\\jelin\\Desktop\\zk\\test\\iphone_15_pro_ground.csv"
    #filepath = "C:\\Users\\jelin\\Desktop\\zk\\test\\iphone_16_pro_ground.csv" #chyba
    #filepath = "C:\\Users\\jelin\\Desktop\\zk\\test\\iphone_se_ground.csv"
    #filepath = "C:\\Users\\jelin\\Desktop\\zk\\test\\pixel_8_pro_ground.csv"

    # Plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    distances_path = os.path.join(current_dir, 'train_dataset','distances.csv')
    distances = pd.read_csv(distances_path, encoding="ISO-8859-1")
    filepath = os.path.join(current_dir,'train_dataset' ,'sim_all_test.csv')
    #filepath = "C:\\Users\\jelin\\Desktop\\zk\\test\\sim_all_test.csv"
    

    data = pd.read_csv(filepath, encoding="ISO-8859-1")  # or encoding="latin1"
    data = data.rename(columns={
        'Frame length on the wire': 'Frame_length',
        'Source': 'Mac',
        'Signal dBm': 'RSSI'
    })
    #data = data[data['Mac'].fillna('').str.startswith('Iphone 14_')]
    #prefixes = ('Iphone 14_a_1', 'Iphone_11_A_3', 'Samsung_A52S_2')  # Add more as needed
    #data = data[data['Mac'].fillna('').str.startswith(prefixes)]
    grouped, tracked_devices = target_tracking(data)
    plot_data(grouped)




    #distances = pd.read_csv('D:\\LIU\\Diploma\\diplom\\code\\distances.csv', encoding="ISO-8859-1")
    values = distances.iloc[:, 0]
    
    # Create masks for conditions
    viable_mask = values < 2
    unlikely_mask = ~viable_mask
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(distances.index[viable_mask], values[viable_mask], color='green', label='Viable Associations', alpha=0.7)
    #plt.scatter(distances.index[unlikely_mask], values[unlikely_mask], color='blue', label='Unlikely Associations', alpha=0.7)
    
    # Labels and legend
    plt.title('Scatter Plot of Distances')
    plt.xlabel('Association')
    plt.ylabel('Distance')
    plt.legend()
    plt.grid(True)
    plt.show()