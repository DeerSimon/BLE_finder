import pandas as pd
import numpy as np
from shapely import wkb
from scipy.optimize import least_squares
from pyproj import Transformer
from rssi_to_distance import rssi_to_distance
import matplotlib.pyplot as plt
from itertools import combinations
import csv
import os
import psycopg2


# Load Data
current_dir = os.path.dirname(os.path.abspath(__file__))
#filepath = os.path.join(current_dir, 'position_est_100.csv')
#
##filepath = "D:\\LIU\\Diploma\\diplom\\code\\position_est_100.csv"
#data = pd.read_csv(filepath)

# Convert WKB to Latitude & Longitude
def wkb_to_latlon(wkb_str):
    if pd.isna(wkb_str):  
        return None, None
    point = wkb.loads(bytes.fromhex(wkb_str))  
    return point.y, point.x  

# Convert lat/lon to UTM
def latlon_to_utm(lat,lon):
    # Define WGS84 (lat/lon) to UTM converter
    wgs84_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)    
    return wgs84_to_utm.transform(lon, lat)  # Returns (x, y) in meters

# Convert UTM to lat/lon
def utm_to_latlon(x, y):
    utm_to_wgs84 = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)
    return utm_to_wgs84.transform(x, y)  # Returns (lon, lat)

def utm_distance(grouped):
    # Convert to UTM Coordinates
    grouped[['utm_x', 'utm_y']] = grouped.apply(lambda row: pd.Series(latlon_to_utm(row['latitude'], row['longitude'])), axis=1)

    # Convert RSSI to Estimated Distance
    grouped['distance'] = grouped['RSSI'].apply(rssi_to_distance)
    return grouped

def trilateration(positions, distances):
    """
    Estimate 2D coordinates from distances to known points (UTM in meters).
    """

    positions = np.array(positions)
    distances = np.array(distances)
    epsilon = 1e-6  # Avoid divide-by-zero

    def weighted_residuals(guess):
        residual = np.linalg.norm(positions - guess, axis=1) - distances
        return residual / (distances + epsilon)  # Inversely weighted residuals

    initial_guess = np.mean(positions, axis=0)
    result = least_squares(weighted_residuals, initial_guess)

    if result.success:
        return tuple(result.x)

    else:
        return (None, None)

def intersection_centroid(positions, distances, resolution=10.0):
    """
    Estimate a centroid based on how many circles (defined by positions & distances) overlap.

    - All 3 overlap → centroid of full intersection
    - Only 2 overlap → centroid of pair, biased toward the smaller-distance one
    - No overlaps → inside best (smallest distance) circle, nudged toward others

    Parameters:
    - positions: list of (x, y) tuples for circle centers.
    - distances: list of radii (same order as positions).
    - resolution: grid resolution (lower = finer accuracy).

    Returns:
    - (x, y): estimated centroid of intersection area.
    """
    positions = np.array(positions)
    distances = np.array(distances)

    def get_overlap_centroid(active_indices):
        pos = positions[active_indices]
        dist = distances[active_indices]

        min_x = np.min(pos[:, 0] - dist)
        max_x = np.max(pos[:, 0] + dist)
        min_y = np.min(pos[:, 1] - dist)
        max_y = np.max(pos[:, 1] + dist)

        x_vals = np.arange(min_x, max_x, resolution)
        y_vals = np.arange(min_y, max_y, resolution)
        xx, yy = np.meshgrid(x_vals, y_vals)
        grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1)

        inside = np.ones(len(grid_points), dtype=bool)
        for i in active_indices:
            dists = np.linalg.norm(grid_points - positions[i], axis=1)
            inside &= dists <= distances[i]

        valid_points = grid_points[inside]
        return np.mean(valid_points, axis=0) if len(valid_points) > 0 else None

    # Try full intersection first
    centroid = get_overlap_centroid([0, 1, 2])
    if centroid is not None:
        return tuple(centroid)

    # Try all 2-circle overlaps, biased toward smaller distance
    for i, j in combinations(range(3), 2):
        centroid = get_overlap_centroid([i, j])
        if centroid is not None:
            # Bias toward smaller-distance circle
            di, dj = distances[i], distances[j]
            wi = 1 / (di + 1e-6)
            wj = 1 / (dj + 1e-6)
            center_i, center_j = positions[i], positions[j]

            # Direction from centroid to stronger signal
            direction = (center_i * wi + center_j * wj) / (wi + wj) - centroid
            direction_norm = direction / (np.linalg.norm(direction) + 1e-6)

            offset = resolution  # small bias step
            biased_centroid = centroid + direction_norm * offset
            return tuple(biased_centroid)

    # Fallback: No overlap — inside best circle, nudged toward others
    best_idx = np.argmin(distances)
    best_pos = positions[best_idx]
    others = [i for i in range(3) if i != best_idx]

    direction = np.mean(positions[others], axis=0) - best_pos
    direction_norm = direction / (np.linalg.norm(direction) + 1e-6)
    offset = 0.25 * distances[best_idx]
    estimated_point = best_pos + direction_norm * offset

    return tuple(estimated_point)

def rolling_trilateration(data,num_of_anchors):
    estimated_points = []
    
    for i in range(len(data) - (num_of_anchors-1)):
        group = data.iloc[i:i+num_of_anchors]
        if len(group) < num_of_anchors:
            continue

        positions = list(zip(group['utm_x'], group['utm_y']))
        distances = list(group['distance'])

        est_x, est_y = trilateration(positions, distances)
        if est_x is not None and est_y is not None:
            estimated_points.append((est_x, est_y))
        #visualize_trilateration(positions, distances, (est_x, est_y))
    return estimated_points

def rolling_trilateration_gap(data, min_x_gap=30, min_y_gap=30):
    estimated_points = []

    used_indices = set()

    for i in range(len(data)):
        candidates = []
        for j in range(i + 1, len(data)):
            for k in range(j + 1, len(data)):
                idxs = [i, j, k]
                if any(idx in used_indices for idx in idxs):
                    continue

                xs = [data.iloc[i]['utm_x'], data.iloc[j]['utm_x'], data.iloc[k]['utm_x']]
                ys = [data.iloc[i]['utm_y'], data.iloc[j]['utm_y'], data.iloc[k]['utm_y']]

                x_spread = max(xs) - min(xs)
                y_spread = max(ys) - min(ys)

                # Require spread on both axes
                if x_spread >= min_x_gap and y_spread >= min_y_gap:
                    candidates = idxs
                    break
            if candidates:
                break

        # Fallback to first available 3 unused indices
        if not candidates:
            candidates = [idx for idx in range(i, len(data)) if idx not in used_indices][:3]
            if len(candidates) < 3:
                continue

        positions = [(data.iloc[idx]['utm_x'], data.iloc[idx]['utm_y']) for idx in candidates]
        distances = [data.iloc[idx]['distance'] for idx in candidates]

        est_x, est_y = trilateration(positions, distances)
        if est_x is not None and est_y is not None:
            estimated_points.append((est_x, est_y))
            used_indices.update(candidates)
            # visualize_trilateration(positions, distances, (est_x, est_y))  # Optional

    return estimated_points

def rolling_centroid(data,num_of_anchors):
    estimated_points = []
    
    for i in range(len(data) - (num_of_anchors-1)):
        group = data.iloc[i:i+num_of_anchors]
        if len(group) < num_of_anchors:
            continue

        positions = list(zip(group['utm_x'], group['utm_y']))
        distances = list(group['distance'])

        est_x, est_y = intersection_centroid(positions, distances)
        if est_x is not None and est_y is not None:
            estimated_points.append((est_x, est_y))
        #visualize_trilateration(positions, distances, (est_x, est_y))
    return estimated_points

def rolling_centroid_gap(data, min_x_gap=30, min_y_gap=30):
    estimated_points = []
    used_indices = set()

    def point_is_separated_enough(p, others, min_x, min_y):
        x_ok = any(abs(p[0] - o[0]) >= min_x for o in others)
        y_ok = any(abs(p[1] - o[1]) >= min_y for o in others)
        return x_ok and y_ok

    def at_least_one_well_separated(points, min_x, min_y):
        for i in range(3):
            p = points[i]
            others = [points[j] for j in range(3) if j != i]
            if point_is_separated_enough(p, others, min_x, min_y):
                return True
        return False

    for i in range(len(data)):
        candidates = []
        for j in range(i + 1, len(data)):
            for k in range(j + 1, len(data)):
                idxs = [i, j, k]
                if any(idx in used_indices for idx in idxs):
                    continue

                positions = [
                    (data.iloc[i]['utm_x'], data.iloc[i]['utm_y']),
                    (data.iloc[j]['utm_x'], data.iloc[j]['utm_y']),
                    (data.iloc[k]['utm_x'], data.iloc[k]['utm_y']),
                ]

                if at_least_one_well_separated(positions, min_x_gap, min_y_gap):
                    candidates = idxs
                    break
            if candidates:
                break

        # Fallback: take the next 3 unused points if no good triplet is found
        if not candidates:
            candidates = [idx for idx in range(i, len(data)) if idx not in used_indices][:3]
            if len(candidates) < 3:
                continue

        positions = [(data.iloc[idx]['utm_x'], data.iloc[idx]['utm_y']) for idx in candidates]
        distances = [data.iloc[idx]['distance'] for idx in candidates]

        est_x, est_y = intersection_centroid(positions, distances)
        if est_x is not None and est_y is not None:
            estimated_points.append((est_x, est_y))
            used_indices.update(candidates)
            # visualize_trilateration(positions, distances, (est_x, est_y))  # Optional

    return estimated_points

def calculate_cep(estimated_points, percentile=50):
    if not estimated_points:
        return None, None, None

    estimated_points = np.array(estimated_points)
    centroid = np.mean(estimated_points, axis=0)
    distances = np.linalg.norm(estimated_points - centroid, axis=1)
    cep_radius = np.percentile(distances, percentile)

    return centroid, cep_radius, distances

def visualize_cep(estimated_points, centroid, cep_radius):
    x_vals, y_vals = zip(*estimated_points)
    fig, ax = plt.subplots()
    ax.scatter(x_vals, y_vals, color='blue', label='Estimated Points')
    ax.scatter(centroid[0], centroid[1], color='red', label='Centroid')

    reference_point = os.path.join(current_dir, 'train_dataset','reference.csv')
    #reference_point =  pd.read_csv("D:\\LIU\\Diploma\\diplom\\code\\reference.csv")
    x,y = latlon_to_utm(reference_point['lat'][1],reference_point['lon'][1]) 


    if reference_point is not None:
        ax.plot(x,y, 'ro', markersize=10, label='Reference Point')
        ax.text(x,y + 0.5, 'Ref.', fontsize=10, color='red')    

    circle = plt.Circle(centroid, cep_radius, color='green', alpha=0.3, label=f'{int(cep_radius)}m CEP')
    ax.add_patch(circle)
    ax.set_aspect('equal', 'box')
    ax.legend()
    #plt.title("Circular Error Probable (CEP)")
    plt.xlabel("UTM X")
    plt.ylabel("UTM Y")
    plt.grid(True)
    plt.show()

def visualize_trilateration(positions, distances, estimated_point_x_3):
    fig, ax = plt.subplots()
    
    positions = np.array(positions)
    distances = np.array(distances)

    # Plot each known position and its distance circle
    for i, (pos, dist) in enumerate(zip(positions, distances)):
        ax.plot(pos[0], pos[1], 'bo')  # Blue dots for known positions
        circle = plt.Circle(pos, dist, color='b', alpha=0.2)#, label=f'Anchor {i+1}' if i == 0 else "")
        ax.add_patch(circle)
        ax.text(pos[0] + 0.5, pos[1] + 0.5, f"P{i+1}", fontsize=10)
    
    reference_point = os.path.join(current_dir, 'train_dataset','reference.csv')
    #reference_point =  pd.read_csv("D:\\LIU\\Diploma\\diplom\\code\\reference.csv")
    x,y = latlon_to_utm(reference_point['lat'][1],reference_point['lon'][1]) 


    if reference_point is not None:
        ax.plot(x,y, 'ro', markersize=10, label='Reference Point')
        ax.text(x,y + 0.5, 'Ref.', fontsize=10, color='red')


    # Plot the estimated point
    if estimated_point_x_3:
        ax.plot(estimated_point_x_3[0], estimated_point_x_3[1], 'rx', markersize=10, label='Estimated Position')
        ax.text(estimated_point_x_3[0] + 0.5, estimated_point_x_3[1] + 0.5, 'Est.', fontsize=10, color='red')

    ax.set_aspect('equal', 'box')
    #ax.set_title('2D Trilateration Visualization')
    ax.legend()
    plt.xlabel("UTM X")
    plt.ylabel("UTM Y")
    plt.grid(True)
    plt.show()

def visualize_centroid(positions, distances, estimated_point):
    fig, ax = plt.subplots()
    
    positions = np.array(positions)
    distances = np.array(distances)

    # Plot each known position and its distance circle
    for i, (pos, dist) in enumerate(zip(positions, distances)):
        ax.plot(pos[0], pos[1], 'bo')  # Blue dots for known positions
        circle = plt.Circle(pos, dist, color='b', alpha=0.2)#, label=f'Anchor {i+1}' if i == 0 else "")
        ax.add_patch(circle)
        ax.text(pos[0] + 0.5, pos[1] + 0.5, f"P{i+1}", fontsize=10)
    
    reference_point = os.path.join(current_dir, 'train_dataset','reference.csv')
    #reference_point =  pd.read_csv("D:\\LIU\\Diploma\\diplom\\code\\reference.csv")
    x,y = latlon_to_utm(reference_point['lat'][1],reference_point['lon'][1]) 


    if reference_point is not None:
        ax.plot(x,y, 'ro', markersize=10, label='Reference Point')
        ax.text(x,y + 0.5, 'Ref.', fontsize=10, color='red')    

    # Plot the estimated point
    if estimated_point:
        ax.plot(estimated_point[0], estimated_point[1], 'rx', markersize=10, label='Estimated Position')
        ax.text(estimated_point[0] + 0.5, estimated_point[1] + 0.5, 'Est.', fontsize=10, color='red')

    ax.set_aspect('equal', 'box')
    #ax.set_title('2D Centroid Visualization')
    ax.legend()
    plt.xlabel("UTM X")
    plt.ylabel("UTM Y")
    plt.grid(True)
    plt.show()

def utm_distance_between(point1, point2):
    """
    Calculate Euclidean distance between two UTM coordinates.
    
    Parameters:
        point1 (tuple or array): (x1, y1) coordinates
        point2 (tuple or array): (x2, y2) coordinates
        
    Returns:
        float: Distance in meters
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1 - point2)
def print_centroid_distance(label, centroid, ref_coords):

    if centroid is not None and all(np.isfinite(centroid)):
        print(f"{label}: {utm_distance_between(centroid, ref_coords):.2f} m")
    else:
        print(f"{label}: Invalid centroid")

def select_spatially_separated_rssi_flexible(data, num=3, min_x_gap=30, min_y_gap=30):
    if len(data) < num:
        return data.sort_values(by='RSSI', ascending=False).head(num)

    sorted_data = data.sort_values(by='RSSI', ascending=False).reset_index(drop=True)

    selected = [sorted_data.iloc[0]]
    used_indices = {0}

    while len(selected) < num:
        best_candidate = None

        for i in range(len(sorted_data)):
            if i in used_indices:
                continue

            candidate = sorted_data.iloc[i]
            good = True

            for selected_point in selected:
                dx = abs(candidate['utm_x'] - selected_point['utm_x'])
                dy = abs(candidate['utm_y'] - selected_point['utm_y'])
                if dx < min_x_gap or dy < min_y_gap:
                    good = False
                    break

            if good:
                best_candidate = candidate
                used_indices.add(i)
                break

        # If no spatially good candidate found, just pick the next unused strongest RSSI
        if best_candidate is None:
            for i in range(len(sorted_data)):
                if i not in used_indices:
                    best_candidate = sorted_data.iloc[i]
                    used_indices.add(i)
                    break

        if best_candidate is not None:
            selected.append(best_candidate)
        else:
            break  # No more points

    return pd.DataFrame(selected)


#def estimate_position(grouped, num_of_anchors):
    #num_of_anchors = 4
    # Estimate Position Using Top 3 RSSI Values per Classification
    estimated_positions = []
    unique_classes = grouped['Classification'].dropna().unique()

    for classification in unique_classes:
        class_data = grouped[grouped['Classification'] == classification]
        #top_3 = class_data.nlargest(num_of_anchors, 'RSSI')  # Select top 3 highest RSSI values
        top_3 = select_spatially_separated_rssi_flexible(class_data, num=num_of_anchors, min_x_gap=10, min_y_gap=10)
        
        #estimated_points = rolling_trilateration(class_data, num_of_anchors)
        #estimated_points_gap = rolling_trilateration_gap(class_data)
        #estimated_points_centroid = rolling_centroid(class_data, num_of_anchors)
        estimated_points_centroid_gap = rolling_centroid_gap(class_data)


    
        if num_of_anchors == 3:
            #centroid, cep_radius, cep_distances = calculate_cep(estimated_points)
            #centroid_c, cep_radius_c, cep_distances_c = calculate_cep(estimated_points_centroid)
            #print_centroid_distance("Centroid of rolling trilateration", centroid, ref_coords)
            #print_centroid_distance("Centroid of rolling CBL", centroid_c, ref_coords)

            #centroid_gap, cep_radius_gap, cep_distances_gap = calculate_cep(estimated_points_gap)
            centroid_c_gap, cep_radius_c_gap, cep_distances_c_gap = calculate_cep(estimated_points_centroid_gap)
            #print_centroid_distance("Centroid of rolling trilateration with Spatial Separation", centroid_gap, ref_coords)
            #print_centroid_distance("Centroid of rolling CBL with Spatial Separation", centroid_c_gap, ref_coords)

        if num_of_anchors == 4:
            #centroid, cep_radius, cep_distances = calculate_cep(estimated_points)
            #centroid_c, cep_radius_c, cep_distances_c = calculate_cep(estimated_points_centroid)
            #print_centroid_distance("Centroid of rolling multilateration", centroid, ref_coords)
            #print_centroid_distance("Centroid of rolling CBL with four measurements", centroid_c, ref_coords)

            #centroid_gap, cep_radius_gap, cep_distances_gap = calculate_cep(estimated_points_gap)
            centroid_c_gap, cep_radius_c_gap, cep_distances_c_gap = calculate_cep(estimated_points_centroid_gap)
            #print_centroid_distance("Centroid of rolling multilateration with Spatial Separation", centroid_gap, ref_coords)
            #print_centroid_distance("Centroid of rolling CBL with Spatial Separation with 4 measurements", centroid_c_gap, ref_coords)

        if len(top_3) == num_of_anchors:
            positions = list(zip(top_3['utm_x'], top_3['utm_y']))
            distances = list(top_3['distance'])

            lat_lon_positions = [utm_to_latlon(x, y) for x, y in positions]

            est_x_3, est_y_3 = trilateration(positions, distances)
            est_x, est_y = intersection_centroid(positions, distances)

            row = {'Classification': classification}

            if est_x is not None and est_y is not None:
                est_lon, est_lat = utm_to_latlon(est_x, est_y)
                est_lon_tri, est_lat_tri = utm_to_latlon(est_x_3, est_y_3)
                row.update({
                    'est_latitude': est_lat,
                    'est_longitude': est_lon,
                    'fingerprint': top_3.iloc[0]['Fingerprint'], #top_3.nlargest(1,'RSSI')['Fingerprint']
                    #'Top1_Lat': lat_lon_positions[0][1], 'Top1_Lon': lat_lon_positions[0][0],
                    #'Top2_Lat': lat_lon_positions[1][1], 'Top2_Lon': lat_lon_positions[1][0],
                    #'Top3_Lat': lat_lon_positions[2][1], 'Top3_Lon': lat_lon_positions[2][0],
                    #'Top1_Distance': distances[0],
                    #'Top2_Distance': distances[1],
                    #'Top3_Distance': distances[2]
                    'est_latitude_trilateration': est_lat_tri,
                    'est_longitude_trilateration': est_lon_tri
                })
                #if num_of_anchors ==3:
                #    print("Trilateration of three highest RSSI measurements: ", utm_distance_between(ref_coords, (est_x_3, est_y_3)), "m")
                #    print("CBL of three highest RSSI measurements: ", utm_distance_between(ref_coords, (est_x, est_y)), "m")
                #if num_of_anchors ==4:
                #    print("Multilateration of four highest RSSI measurements: ", utm_distance_between(ref_coords, (est_x_3, est_y_3)), "m")
                #    print("CBL of four highest RSSI measurements: ", utm_distance_between(ref_coords, (est_x, est_y)), "m")                    
            #if centroid is not None:
            #    est_centroid_lon, est_centroid_lat = utm_to_latlon(centroid_c[0], centroid_c[1])
            #    row.update({
            #        'Centroid_Lat': est_centroid_lat,
            #        'Centroid_Lon': est_centroid_lon,
            #        'CEP_Radius': cep_radius_c
            #    })
            #else:
            #    print(f"Warning: centroid is None for classification {classification}")

            estimated_positions.append(row)
            #visualize_trilateration(positions, distances,estimated_point_x_3=(est_x_3, est_y_3))
            #print("Trilateration of 3 points: ",utm_distance_between(ref_coords, (est_x_3, est_y_3)), "m")
            #tri=(est_x_3, est_y_3)
            #visualize_centroid(positions, distances,estimated_point=(est_x, est_y))
            #cent=(est_x, est_y)
            #print("CBL of 3 points: ",utm_distance_between(ref_coords, (est_x, est_y)), "m")
        
        if centroid_c_gap is not None and np.all(np.isfinite(centroid_c_gap)):
            lat_lon = utm_to_latlon(centroid_c_gap[0], centroid_c_gap[1])

            # Get fingerprint (use the one with highest RSSI instead of first row)
            fingerprint = top_3.nlargest(1, 'RSSI')['Fingerprint'].iloc[0]

            # Read existing rows if the file exists
            output_file = os.path.join(current_dir, 'output.csv')
            if os.path.exists(output_file):
                with open(output_file, 'r', newline='') as csvfile:
                    reader = list(csv.reader(csvfile))
            else:
                reader = [["Classification", "Latitude", "Longitude", "Fingerprint"]]

            # Convert rows to a dict keyed by Classification
            header = reader[0]

            # Handle both old format (3 columns) and new format (4 columns)
            rows = {}
            for row in reader[1:]:
                if len(row) >= 3:  # At least Classification, Lat, Lon
                    if len(row) == 3:  # Old format, add empty fingerprint
                        rows[row[0]] = [row[1], row[2], ""]
                    else:  # New format with fingerprint
                        rows[row[0]] = row[1:]

            # Update or insert the classification with fingerprint
            rows[classification] = [lat_lon[1], lat_lon[0], fingerprint]  # lat_lon = (lon, lat)

            # Ensure header has all 4 columns
            if len(header) < 4:
                header = ["Classification", "Latitude", "Longitude", "Fingerprint"]

            # Write the updated data back
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                for cls, coords in rows.items():
                    writer.writerow([cls] + coords)

            print(f"✓ Added: Classification {classification}, Fingerprint: {fingerprint}")

        else:
            print("Invalid centroid_c_gap, row not added.")


    return estimated_positions  #, tri, cent, estimated_points, estimated_points_centroid, centroid,centroid_c, centroid_gap, centroid_c_gap



import psycopg2
import csv
import os
import numpy as np

def estimate_position(grouped, num_of_anchors):
    # Database configuration
    db_config = {
        'database': 'rasp',
        'user': 'simje951',
        'password': 'xA9f8E7G6emt',
        'host': '192.168.1.10',
        'port': '5432'
    }
    
    # Estimate Position Using Top anchors RSSI Values per Classification
    estimated_positions = []
    unique_classes = grouped['Classification'].dropna().unique()

    # Database connection
    conn = None
    cursor = None
    
    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        for classification in unique_classes:
            class_data = grouped[grouped['Classification'] == classification]
            top_3 = select_spatially_separated_rssi_flexible(class_data, num=num_of_anchors, min_x_gap=10, min_y_gap=10)
            
            estimated_points_centroid_gap = rolling_centroid_gap(class_data)

            if num_of_anchors == 3:
                centroid_c_gap, cep_radius_c_gap, cep_distances_c_gap = calculate_cep(estimated_points_centroid_gap)

            if num_of_anchors == 4:
                centroid_c_gap, cep_radius_c_gap, cep_distances_c_gap = calculate_cep(estimated_points_centroid_gap)

            if len(top_3) == num_of_anchors:
                positions = list(zip(top_3['utm_x'], top_3['utm_y']))
                distances = list(top_3['distance'])

                lat_lon_positions = [utm_to_latlon(x, y) for x, y in positions]

                est_x_3, est_y_3 = trilateration(positions, distances)
                est_x, est_y = intersection_centroid(positions, distances)

                row = {'Classification': classification}

                if est_x is not None and est_y is not None:
                    est_lon, est_lat = utm_to_latlon(est_x, est_y)
                    est_lon_tri, est_lat_tri = utm_to_latlon(est_x_3, est_y_3)
                    fingerprint = top_3.iloc[0]['Fingerprint']
                    
                    row.update({
                        'est_latitude': est_lat,
                        'est_longitude': est_lon,
                        'fingerprint': fingerprint,
                        'est_latitude_trilateration': est_lat_tri,
                        'est_longitude_trilateration': est_lon_tri
                    })

                    estimated_positions.append(row)
            
            # Database insertion with geometry handling
            if centroid_c_gap is not None and np.all(np.isfinite(centroid_c_gap)):
                lat_lon = utm_to_latlon(centroid_c_gap[0], centroid_c_gap[1])
                fingerprint = top_3.nlargest(1, 'RSSI')['Fingerprint'].iloc[0]
                
                # Prepare geometry values for database insertion
                # CBL geometry (from intersection_centroid method)
                if est_x is not None and est_y is not None:
                    cbl_geometry = f"POINT({est_lon} {est_lat})"
                else:
                    cbl_geometry = None
                
                # CBL Rolling geometry (from rolling centroid gap)
                cbl_rolling_geometry = f"POINT({lat_lon[0]} {lat_lon[1]})"  # lat_lon = (lon, lat)
                
                # Trilateration geometry
                if est_x_3 is not None and est_y_3 is not None:
                    trilateration_geometry = f"POINT({est_lon_tri} {est_lat_tri})"
                else:
                    trilateration_geometry = None

                # Insert into database
                try:
                    # Simple insert for each classification
                    insert_query = """
                    INSERT INTO search_area.location_estimates 
                    (classification, fingerprint, cbl, cbl_rolling, trilateration)
                    VALUES (%s, %s, ST_GeomFromText(%s, 4326), ST_GeomFromText(%s, 4326), ST_GeomFromText(%s, 4326))
                    """
                    
                    cursor.execute(insert_query, (
                        int(classification),
                        fingerprint,
                        cbl_geometry,
                        cbl_rolling_geometry,
                        trilateration_geometry
                    ))
                    
                    print(f"✓ Database: Added Classification {classification}, Fingerprint: {fingerprint}")
                    
                except psycopg2.Error as e:
                    print(f"✗ Database error for classification {classification}: {e}")
                    conn.rollback()
                    continue

                # CSV file handling (keeping original functionality)
                output_file = os.path.join(current_dir, 'output.csv')
                if os.path.exists(output_file):
                    with open(output_file, 'r', newline='') as csvfile:
                        reader = list(csv.reader(csvfile))
                else:
                    reader = [["Classification", "Latitude", "Longitude", "Fingerprint"]]

                # Convert rows to a dict keyed by Classification
                header = reader[0]

                # Handle both old format (3 columns) and new format (4 columns)
                rows = {}
                for row in reader[1:]:
                    if len(row) >= 3:  # At least Classification, Lat, Lon
                        if len(row) == 3:  # Old format, add empty fingerprint
                            rows[row[0]] = [row[1], row[2], ""]
                        else:  # New format with fingerprint
                            rows[row[0]] = row[1:]

                # Update or insert the classification with fingerprint
                rows[classification] = [lat_lon[1], lat_lon[0], fingerprint]  # lat_lon = (lon, lat)

                # Ensure header has all 4 columns
                if len(header) < 4:
                    header = ["Classification", "Latitude", "Longitude", "Fingerprint"]

                # Write the updated data back
                with open(output_file, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(header)
                    for cls, coords in rows.items():
                        writer.writerow([cls] + coords)

                print(f"✓ CSV: Added Classification {classification}, Fingerprint: {fingerprint}")

            else:
                print("Invalid centroid_c_gap, row not added.")

        # Commit all database changes
        conn.commit()
        print("✓ All database transactions committed successfully")
        
    except psycopg2.Error as e:
        print(f"✗ Database connection error: {e}")
        if conn:
            conn.rollback()
    
    finally:
        # Close database connections
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    return estimated_positions


reference_point_path = os.path.join(current_dir, 'train_dataset','reference.csv')
reference_point = pd.read_csv(reference_point_path)
#reference_point =  pd.read_csv("D:\\LIU\\Diploma\\diplom\\code\\reference.csv")
#reference_point =  pd.read_csv("D:\\LIU\\Diploma\\diplom\\code\\reference.csv")
#reference_point =  pd.read_csv("D:\\LIU\\Diploma\\diplom\\code\\search_3\\ref_search_3.csv")
#reference_point = pd.read_csv("D:\\LIU\\Diploma\\diplom\\code\\reference_afrika.csv")
x,y = latlon_to_utm(reference_point['lat'][0],reference_point['lon'][0])   #search 3
ref_coords = (x, y)



