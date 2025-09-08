import psycopg2
import pandas as pd
import numpy as np
import csv
from shapely import wkb
from target_tracking import target_tracking,plot_data 
from position_estimation import wkb_to_latlon, utm_distance, estimate_position, utm_to_latlon
from attributes import group_data
import os

# Database credentials
DATABASE = 'rasp'
USER = 'simje951'
PASSWORD = 'xA9f8E7G6emt'
HOST = 'localhost'
PORT = '5432'

def process_bluetooth_data(input_data, time_window=5, num_of_anchors=4, position=1):
    """
    Main function to process bluetooth data
    
    Args:
        input_data: pandas DataFrame with bluetooth data
        time_window: time window size in seconds (default: 5)
        num_of_anchors: number of anchors for position estimation (default: 4)
        position: whether to calculate positions (default: 1)
    
    Returns:
        tuple: (grouped, grouped_2, tracked_devices, est_positions, data)
    """
    # Use input data instead of loading from file
    data = input_data.copy()
    
    # Column renaming
    data = data.rename(columns={
        'row_id': 'No.',
        'rssi': 'RSSI',
        'fingerprint': 'Company ID',
        'frame_length': 'Frame_length',
        'mac': 'Mac',
        'occurrences': 'Occurrences',
        'time': 'Time'
    })
    
    # Handle location data
    if 'location' in data.columns:
        data = data.dropna(subset=['location'])
        data[['latitude', 'longitude']] = data['location'].apply(lambda wkb_str: pd.Series(wkb_to_latlon(wkb_str)))

    # TARGET TRACKING
    grouped, tracked_devices = target_tracking(data)

    if 'lat' in data.columns and 'lon' in data.columns:
        data = data.rename(columns={'lat': 'latitude', 'lon': 'longitude'})

    # Time window processing
    data['Time Window'] = np.floor(data['Time'] / time_window) * time_window

    grouped_2 = group_data(data, time_window, 'Time Window')
    grouped_2 = grouped_2.dropna(subset=['latitude'])

    est_positions = []
    
    if position == 1:
        # Drop duplicates to keep the first occurrence
        unique_classification = grouped.drop_duplicates(subset='Mac')[['Mac', 'Classification']]

        # Now map with a unique index
        grouped_2['Classification'] = grouped_2['Mac'].map(unique_classification.set_index('Mac')['Classification'])
        grouped_2 = grouped_2.dropna(subset=['Classification'])
        
        grouped_2 = utm_distance(grouped_2)

        est_positions = estimate_position(grouped_2, num_of_anchors)
        save_results_to_files(est_positions, output_dir=None)

    # Return results as tuple for easy unpacking
    return grouped, grouped_2, tracked_devices, est_positions, data

def save_results_to_files(est_positions,output_dir=None):
    """
    Save processing results to files
    
    Args:
        grouped: target tracking results
        grouped_2: grouped data with classifications
        tracked_devices: list of tracked devices
        est_positions: estimated positions
        data: processed data
        output_dir: output directory (default: current script directory)
    """
    if output_dir is None:
        output_dir = os.path.dirname(__file__)
    
    # Save estimated positions
    if est_positions:
        csv_file_path = os.path.join(output_dir, 'est_positions.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            if isinstance(est_positions, list) and len(est_positions) > 0:
                fieldnames = est_positions[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(est_positions)
        print(f"CSV saved to: {csv_file_path}")
    

# Original script logic for standalone execution
if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(__file__)
    filepath = os.path.join(current_dir, 'wax_truth.csv')
    
    # Build the file path - you can change this to your specific dataset
    #filepath = os.path.join(current_dir, 'afrika', 'data.csv')
    #filepath = "D:\\LIU\\Diploma\\diplom\\code\\position_est_100.csv"
    #filepath = "D:\\LIU\\Diploma\\diplom\\code\\wax_truth.csv"  #VAXHOLM
    #filepath = "D:\\LIU\\Diploma\\diplom\\code\\search_1\\data_1.csv"
    filepath = "D:\\LIU\\Diploma\\diplom\\code\\search_3\\data_3_cut.csv" #Kalmarden
    #filepath = "D:\\LIU\\Diploma\\diplom\\code\\afrika\\data.csv" #South Africa
    #filepath = "D:\\LIU\\Diploma\\diplom\\code\\Strömsfors\\data_15_16.csv" #round 1
    #filepath = "D:\\LIU\\Diploma\\diplom\\code\\Strömsfors\\data_16_17.csv" #round 2
    #filepath = "D:\\LIU\\Diploma\\diplom\\code\\Strömsfors\\data_16_17_05_drone.csv" #round 2
    
    try:
        data = pd.read_csv(filepath)
        
        # Optional data filtering
        #data = data[data['lat'].notna()].reset_index(drop=True)
        #data = data[data['Time'].between(1930 ,2200)].reset_index(drop=True) #afrika
        #data = data[data['Time'].between(13400 ,14850)].reset_index(drop=True) #round 1 15:08-15:44
        #data = data[data['Time'].between(16320 ,16800)].reset_index(drop=True) #round 2 16:08-16:38
        #data = data[data['Time'].between(17000 ,17460)].reset_index(drop=True) #round 3 16:54-17:02
        
        # Process the data
        grouped, grouped_2, tracked_devices, est_positions, data = process_bluetooth_data(data, time_window=5, num_of_anchors=4, position=1)
        
        # Save results to files
        save_results_to_files(est_positions, output_dir=current_dir)
        
        # Optional: plot data
        #plot_data(grouped)
        
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {e}")