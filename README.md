# BLE Finder

A sophisticated Bluetooth Low Energy (BLE) device tracking and positioning system that processes bluetooth scan data to identify, track, and estimate the positions of mobile devices using advanced signal processing and trilateration techniques.

## Description

BLE Finder is a comprehensive solution for tracking and locating Bluetooth devices using RSSI (Received Signal Strength Indicator) measurements from multiple anchor points. The system combines advanced target tracking algorithms with position estimation techniques to provide real-time device location capabilities.

### Key Features

- **Real-time Device Tracking**: Identifies and tracks multiple BLE devices simultaneously using MAC address pattern analysis
- **Position Estimation**: Uses trilateration and Circle-Based Localization (CBL) algorithms to estimate device positions
- **Interactive GUI**: User-friendly interface for real-time simulation and visualization
- **Database Integration**: PostgreSQL integration for data storage and retrieval
- **Advanced Signal Processing**: Implements sophisticated algorithms for noise reduction and accuracy improvement
- **Persistent State Management**: Maintains device tracking across sessions using JSON metadata
- **Visualization Tools**: Real-time plotting of device counts, position estimates, and tracking results

## Installation

### Prerequisites

- Python 3.8 or higher
- PostgreSQL database
- Required Python packages (see requirements below)

### Dependencies

Install the required packages:

```bash
pip install pandas numpy matplotlib scipy shapely pyproj psycopg2-binary tkinter scikit-learn
```

### Database Setup

1. Install PostgreSQL
2. Create a database named 'rasp'
3. Update database credentials in the configuration files:
   ```python
   db_config = {
       'database': 'rasp',
       'user': 'your_username',
       'password': 'your_password',
       'host': 'your_host',
       'port': '5432'
   }
   ```

## Usage

### GUI Application

Launch the interactive GUI for real-time simulation:

```bash
python GUI.py
```

The GUI provides:
- **Time Controls**: Set start and end times for data processing
- **Data Collection Modes**: Choose between lookback window or cumulative data processing
- **Real-time Statistics**: View tracked devices, position estimates, and processing status
- **Interactive Plots**: Visualize device counts and estimates over time
- **Control Buttons**: Start, stop, and reset simulation functions

### Command Line Processing

Process bluetooth data directly:

```python
from main import process_bluetooth_data
import pandas as pd

# Load your data
data = pd.read_csv('your_bluetooth_data.csv')

# Process the data
grouped, grouped_2, tracked_devices, est_positions, processed_data = process_bluetooth_data(
    data, 
    time_window=5, 
    num_of_anchors=3, 
    position=1
)
```

### Data Format

Input data should be a CSV file with the following columns:
- `Time`: Timestamp of the measurement
- `Mac`: MAC address of the detected device
- `RSSI`: Signal strength in dBm
- `Frame_length`: Length of the bluetooth frame
- `latitude`/`longitude`: Location coordinates of the anchor point
- `Fingerprint`: Device fingerprint information

## Core Components

### 1. Target Tracking (`target_tracking.py`)

Implements advanced device identification and tracking:
- **MAC Address Analysis**: Identifies devices based on communication patterns
- **Temporal Tracking**: Maintains device state across time windows
- **Classification System**: Assigns unique IDs to tracked devices
- **Persistence**: Saves/loads tracking state using JSON metadata

### 2. Position Estimation (`position_estimation.py`)

Provides multiple positioning algorithms:
- **Trilateration**: Classic geometric approach using 3+ anchor points
- **Circle-Based Localization (CBL)**: Advanced intersection-based positioning
- **Rolling Algorithms**: Processes multiple measurement sets for improved accuracy
- **Spatial Separation**: Ensures anchor point diversity for better positioning
- **UTM Coordinate System**: Uses Universal Transverse Mercator for precise calculations

### 3. Main Processing (`main.py`)

Orchestrates the complete processing pipeline:
- Data preprocessing and cleaning
- Integration of tracking and positioning modules
- Result compilation and export
- Database integration for persistent storage

### 4. GUI Controller (`GUI.py`)

Provides interactive interface:
- **Real-time Simulation**: Process historical data with configurable parameters
- **Visualization**: Live plots of tracking and positioning results
- **Database Integration**: Direct connection to PostgreSQL for data retrieval
- **Export Functionality**: Save results to CSV files

## Algorithms

### Target Tracking Algorithm

1. **Data Preprocessing**: Filters scan requests and malformed packets
2. **Temporal Grouping**: Groups measurements by time windows
3. **Device Fingerprinting**: Creates unique signatures based on frame patterns
4. **Association Logic**: Links measurements to existing or new device tracks
5. **State Management**: Updates device states and handles timeouts

### Position Estimation Methods

1. **Trilateration**: Solves geometric equations using least squares optimization
2. **Circle-Based Localization**: Finds intersection centroids of signal circles
3. **Rolling Processing**: Applies algorithms across multiple measurement sets
4. **Spatial Optimization**: Selects geometrically diverse anchor points

## Configuration

### Time Windows
- Default processing window: 5 seconds
- Configurable update intervals
- Adjustable lookback periods

### Positioning Parameters
- Number of anchors: 3-4 recommended
- Minimum spatial separation: 30m (configurable)
- RSSI to distance conversion parameters

### Database Schema

The system expects a PostgreSQL database with tables containing:
- Bluetooth scan data with timestamps
- Anchor location information (stored as PostGIS geometry)
- Device classification and positioning results

## Output

The system generates several types of output:

1. **CSV Files**:
   - `est_positions.csv`: Estimated device positions
   - `output.csv`: Classification results with coordinates
   - `grouped_data.csv`: Processed tracking data

2. **Database Records**: Persistent storage in PostgreSQL
3. **Real-time Visualizations**: Interactive plots and statistics
4. **Log Files**: Processing status and debugging information

## Performance Considerations

- **Memory Usage**: Scales with the number of tracked devices and time window size
- **Processing Speed**: Optimized for real-time operation with configurable intervals
- **Database Performance**: Uses indexed queries for efficient data retrieval
- **Accuracy**: Position accuracy depends on anchor geometry and signal quality

## Troubleshooting

### Common Issues

1. **Database Connection Errors**: Verify PostgreSQL is running and credentials are correct
2. **Missing Dependencies**: Install all required Python packages
3. **Data Format Issues**: Ensure CSV files have the correct column names
4. **Memory Issues**: Reduce time window size or implement data chunking

### Debug Mode

Enable detailed logging by modifying the log level in the configuration files.

## Contributing

This project is part of ongoing research in BLE positioning systems. Contributions are welcome in the following areas:

- Algorithm improvements for positioning accuracy
- GUI enhancements and visualization features
- Performance optimizations
- Additional positioning algorithms
- Test case development

## License

This project is developed for research purposes. Please contact the authors for usage permissions and licensing information.
```bash
jelinek.simon99@gmail.com
```

## Authors and Acknowledgment

Developed as part of research in Bluetooth Low Energy positioning systems. Special thanks to contributors and the research community for algorithm development and testing.

## Project Status

Active development - The system is functional and being continuously improved for enhanced accuracy and performance in real-world deployments.