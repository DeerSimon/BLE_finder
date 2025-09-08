from assign_meas import assign_measurements
def process_measurements(time, prev_meas, tracked_devices, device_updates, results, device_counter,time_window):
    """
    Processes the measurements and updates tracking information.
    """
    if len(prev_meas) > 0:
        assignments = assign_measurements(prev_meas, tracked_devices,time, time_window)
        
        for meas, matched_mac in assignments:
            matched_device_id = tracked_devices[matched_mac]['id']
            del tracked_devices[matched_mac]
            device_updates.append({
                'new': meas['mac'], 
                'old': matched_mac,
                'id': matched_device_id,
                'idx': meas['idx']})
            
            tracked_devices[meas['mac']] = {
                'id': matched_device_id,
                'signal': meas['signal'],
                'frame_length': meas['frame_length'],                
                'mac period': meas['mac period'],
                'fingerprint': meas['fingerprint'],
                'occurrences': meas['occurrences'],
                'last_seen': meas['last_seen'],
            }
            results[meas['idx']] = matched_device_id
        
        # Handle unmatched measurements (potential new devices)
        for i, (meas_mac,meas) in enumerate(prev_meas.items()):
            if meas_mac not in [assignment[0]['mac'] for assignment in assignments]:
                tracked_devices[meas_mac] = {
                    'id': device_counter,
                    'signal': meas['signal'],
                    'frame_length': meas['frame_length'],                
                    'mac period': meas['mac period'],
                    'fingerprint': meas['fingerprint'],
                    'occurrences': meas['occurrences'],
                    'last_seen': meas['last_seen'],
                }
                results[meas['idx']] = device_counter
                device_counter += 1
        
        #prev_meas.clear()
    
    return tracked_devices, device_counter