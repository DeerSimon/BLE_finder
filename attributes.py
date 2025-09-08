#def group_data(data, time_window, attr):
#    grouped = data.groupby([attr, 'Mac']).agg({
#        'RSSI': 'median',
#        'Frame_length': 'mean',
#        #'Time': lambda x: x.diff().where(lambda d: d > 0.01).dropna().nsmallest(5).mean() if len(x) > 1 else 0,
#        #'Time': lambda x: x.diff().where(lambda d: d < 10.71).max() if len(x) > 1 else 0,
#        #'Time': lambda x: x.diff().where(lambda d: d > 0.01).min() if len(x) > 1 else 0,
#        'Time': lambda x: x.diff().min() if len(x) > 1 else 10,
#        'Mac': lambda x: x.count() / time_window,
#        'Company ID': lambda x: x.fillna('a').sort_values().iloc[-1],
#        'No.': 'min'  # Adding the aggregation for No.
#    }).rename(columns={'Time': 'MAC Period', 'Mac': 'Occurrences', 'Company ID': 'Fingerprint'}).reset_index()
#    
#    grouped = grouped.sort_values(by=['No.'], ascending=True)
#    
#    return grouped

import pandas as pd
import numpy as np
MAX = 10.6
def group_data(data, time_window, attr):
    if 'latitude' in data.columns and 'longitude' in data.columns and 'gnss_altitude' in data.columns:
        grouped = data.groupby([attr, 'Mac']).agg({
            'RSSI': 'median',
            'Frame_length': 'mean',
            #'Time': lambda x: x.diff().min() if len(x) > 1 else 10,
            #'Time': lambda x: x.diff().where(lambda d: d > 0.01).min() if len(x) > 1 else 10,´
            'Time': lambda x: min_val if (min_val := x.diff().where(lambda d: d > 0.01).min()) <= MAX else MAX if len(x) > 1 else MAX,
            'Mac': lambda x: x.count() / time_window,
            'Company ID': lambda x: x.fillna('a').sort_values().iloc[-1],
            'No.': 'min',
            'latitude': 'mean',
            'longitude': 'mean',
            'gnss_altitude': 'mean'
        }).rename(columns={'Time': 'MAC Period', 'Mac': 'Occurrences', 'Company ID': 'Fingerprint'}).reset_index()
    else:
        grouped = data.groupby([attr, 'Mac']).agg({
            'RSSI': 'median',
            'Frame_length': 'mean',
            #'Time': lambda x: x.diff().where(lambda d: d > 0.01).min() if len(x) > 1 else 10,
            'Time': lambda x: min_val if (min_val := x.diff().where(lambda d: d > 0.01).min()) <= MAX else MAX if len(x) > 1 else MAX,
            #'Time': lambda x: x.diff().min() if len(x) > 1 else 10,
            'Mac': lambda x: x.count() / time_window,
            'Company ID': lambda x: x.fillna('a').sort_values().iloc[-1],
            'No.': 'min'
        }).rename(columns={'Time': 'MAC Period', 'Mac': 'Occurrences', 'Company ID': 'Fingerprint'}).reset_index()

    grouped = grouped.sort_values(by=['No.'], ascending=True)
    
    return grouped
def process_grouped_data(filtered_data, time_slice):
    """
    Process grouped data by calculating median RSSI, max time difference, 
    occurrences, and fingerprint.

    Parameters:
    filtered_data (DataFrame): Input data to be processed
    time_slice (int): Time slice value used in calculation

    Returns:
    DataFrame: Processed data
    """
    #grouped_targets = filtered_data.groupby(['Mac', 'Frame_length']).agg({
    #    'RSSI': 'median',
    #    #'Time': lambda x: x.diff().where(lambda d: d > 0.01).dropna().nsmallest(5).mean() if len(x) > 1 else 0,
    #    #'Time': lambda x: x.diff().where(lambda d: d < 10.71).max() if len(x) > 1 else 0,
    #    'Time': lambda x: x.diff().min() if len(x) > 1 else 10,
    #    'Mac': lambda x: x.count() / time_slice,
    #    'Company ID': lambda x: x.fillna('a').sort_values().iloc[-1]
    #}).rename(columns={'Time': 'MAC Period', 'Mac': 'Occurrences', 'Company ID': 'Fingerprint'}).reset_index()
    if 'latitude' in filtered_data.columns and 'longitude' in filtered_data.columns and 'gnss_altitude' in filtered_data.columns:
        grouped_targets = filtered_data.groupby(['Mac', 'Frame_length']).agg({
            'RSSI': 'median',
            #'Time': lambda x: x.diff().min() if len(x) > 1 else 10,
            #'Time': lambda x: x.diff().where(lambda d: d > 0.01).min() if len(x) > 1 else 10,
            'Time': lambda x: min_val if (min_val := x.diff().where(lambda d: d > 0.01).min()) <= MAX else MAX if len(x) > 1 else MAX,
            'Mac': lambda x: x.count() / time_slice,
            'Company ID': lambda x: x.fillna('a').sort_values().iloc[-1],
            'latitude': 'mean',
            'longitude': 'mean',
            'gnss_altitude': 'mean'
        }).rename(columns={'Time': 'MAC Period', 'Mac': 'Occurrences', 'Company ID': 'Fingerprint'}).reset_index()
    else:
        grouped_targets = filtered_data.groupby(['Mac', 'Frame_length']).agg({
            'RSSI': 'median',
            #'Time': lambda x: x.diff().where(lambda d: d > 0.01).min() if len(x) > 1 else 10,
            'Time': lambda x: min_val if (min_val := x.diff().where(lambda d: d > 0.01).min()) <= MAX else MAX if len(x) > 1 else MAX,
            #'Time': lambda x: x.diff().min() if len(x) > 1 else 10,
            'Mac': lambda x: x.count() / time_slice,
            'Company ID': lambda x: x.fillna('a').sort_values().iloc[-1]
        }).rename(columns={'Time': 'MAC Period', 'Mac': 'Occurrences', 'Company ID': 'Fingerprint'}).reset_index()

    return grouped_targets


def most_frequent_frame_length_per_mac(data):
    most_frequent_frame = (
        data.groupby(['Mac', 'Frame_length'])
        .size()
        .reset_index(name='count')
        .sort_values(['Mac', 'count', 'Frame_length'], ascending=[True, False, True])
        .drop_duplicates(subset='Mac', keep='first')
        .drop(columns=['count'])
    )
    # Merge to keep only the most frequent Frame_length per Mac
    data = data.merge(most_frequent_frame, on=['Mac', 'Frame_length'])
    return data
