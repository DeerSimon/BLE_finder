def find_mac_path(mac, device_updates, path=None, seen=None):
    if path is None:
        path = []
    if seen is None:
        seen = set()

    if mac in seen:
        return path  # Prevent infinite loops in cyclic cases

    seen.add(mac)
    path.append(mac)  # Track the current MAC in the path

    # Find the first matching update
    matching_updates = [update for update in device_updates if update['old'] == mac]
    
    if matching_updates:
        new_mac = matching_updates[0]['new']
        return find_mac_path(new_mac, device_updates, path, seen)  # Recursive call
    
    return path  # Return full path