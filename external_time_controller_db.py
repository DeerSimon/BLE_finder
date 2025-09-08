import pandas as pd
import numpy as np
import time
import os
import psycopg2
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import csv

# Import your main processing script
from main import process_bluetooth_data, save_results_to_files

class DatabaseBluetoothController:
    """
    Database-driven controller for real-time bluetooth data processing
    """
    
    def __init__(self, db_config, start_time, end_time, update_interval_seconds=10, lookback_window_seconds=60):
        """
        Initialize the database controller
        
        Args:
            db_config: dictionary with database connection parameters
            start_time: datetime object for simulation start
            end_time: datetime object for simulation end
            update_interval_seconds: interval between database queries in seconds
            lookback_window_seconds: how far back to keep data in seconds
        """
        self.db_config = db_config
        self.update_interval = update_interval_seconds
        self.lookback_window = lookback_window_seconds
        
        # Time tracking
        self.start_time = start_time
        self.end_time = end_time       
        self.current_time = self.start_time
        
        # Data storage
        self.accumulated_data = pd.DataFrame()
        
        # Current state variables
        self.current_tracked_devices = []
        self.current_estimated_positions = []
        self.current_grouped_data = pd.DataFrame()
        
        # Results tracking
        self.iteration_count = 0
        self.all_results = []
        
        # Test database connection
        self._test_connection()
        
        print(f"Database controller initialized!")
        print(f"Time range: {self.start_time} - {self.end_time}")
        print(f"Update interval: {self.update_interval} seconds")
        print(f"Lookback window: {self.lookback_window} seconds")
        print("-" * 60)
    
    def _test_connection(self):
        """Test database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            print(f"✓ Database connection successful!")
            print(f"  PostgreSQL version: {version[0]}")
            cursor.close()
            conn.close()
        except Exception as e:
            raise ConnectionError(f"Database connection failed: {e}")
    
    def query_database_window(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Query database for data in specific time window
        
        Args:
            start_time: start of time window
            end_time: end of time window
            
        Returns:
            DataFrame with queried data
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            
            query = """
            SELECT * FROM sensor.test 
            WHERE protocol = 'BLE' 
            AND created_at >= %s 
            AND created_at < %s
            ORDER BY created_at
            """
            
            # Execute query
            df = pd.read_sql_query(query, conn, params=[start_time, end_time])
            
            print(df)
            conn.close()
            
            print(f"  📊 Queried {len(df)} records from database")
            
            return df
            
        except Exception as e:
            print(f"  ❌ Database query error: {e}")
            return pd.DataFrame()
    
    def update_accumulated_data(self, new_data: pd.DataFrame):
        """
        Update accumulated data with new data and apply lookback window
        
        Args:
            new_data: new data from database query
        """
        if len(new_data) > 0:
            # Add new data
            self.accumulated_data = pd.concat([self.accumulated_data, new_data], ignore_index=True)
            
            # Apply lookback window based on created_at
            cutoff_time = self.current_time - timedelta(seconds=self.lookback_window)
            
            if 'created_at' in self.accumulated_data.columns:
                # Convert to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(self.accumulated_data['created_at']):
                    self.accumulated_data['created_at'] = pd.to_datetime(self.accumulated_data['created_at'])
                
                # Filter by lookback window
                #self.accumulated_data = self.accumulated_data[
                #    self.accumulated_data['created_at'] >= cutoff_time
                #].reset_index(drop=True)
    
    def process_current_iteration(self):
        """Process current time iteration"""
        self.iteration_count += 1
        window_end = self.current_time + timedelta(seconds=self.update_interval)
        
        if self.iteration_count == 10:
            print("🚀 10th iteration reached! Stopping simulation.")
            
        print(f"\n=== Iteration {self.iteration_count} ===")
        print(f"Time window: {self.current_time.strftime('%H:%M:%S')} - {window_end.strftime('%H:%M:%S')}")
        
        # Query database for new data
        new_data = self.query_database_window(self.current_time, window_end)
        
        # Update accumulated data
        self.update_accumulated_data(new_data)
        print(f"  📈 Accumulated records: {len(self.accumulated_data)}")
        
        # Process data if we have any
        if len(self.accumulated_data) > 0:
            try:
                # Prepare data for processing (convert created_at to relative seconds)
                processed_data = self.prepare_data_for_processing(self.accumulated_data.copy())
                
                # Call main processing function
                grouped, grouped_2, tracked_devices, est_positions, final_data = process_bluetooth_data(
                    processed_data,
                    time_window=5,
                    num_of_anchors=3,
                    position=1
                )
                
                # Update current state variables
                self.current_tracked_devices = tracked_devices
                self.current_estimated_positions = est_positions
                self.current_grouped_data = grouped_2
                
                print(f"  🎯 Current tracked devices: {len(self.current_tracked_devices)}")
                print(f"  📍 Current estimated positions: {len(self.current_estimated_positions)}")
                print(f"  📊 Current grouped data shape: {self.current_grouped_data.shape}")
                
                # Print some details about tracked devices
                if len(self.current_tracked_devices) > 0:
                    unique_classes = grouped['Classification'].unique() if len(grouped) > 0 else []
                    print(f"  🏷️  Device classifications: {unique_classes}")
                
                # Print position estimates if available
                if len(self.current_estimated_positions) > 0:
                    print("  🗺️  Position estimates:")
                    for i, pos in enumerate(self.current_estimated_positions[:3]):  # Show first 3
                        if isinstance(pos, dict):
                            lat = pos.get('est_latitude', 'N/A')
                            lon = pos.get('est_longitude', 'N/A')
                            classification = pos.get('Classification', 'N/A')
                            print(f"    Device {classification}: ({lat}, {lon})")
                
                # Save results to Excel
                #self.save_iteration_results()
                
                # Store results for history
                iteration_result = {
                    'iteration': self.iteration_count,
                    'time_window': (self.current_time, window_end),
                    'new_records': len(new_data),
                    'accumulated_records': len(self.accumulated_data),
                    'tracked_devices': len(tracked_devices),
                    'position_estimates': len(est_positions),
                    'grouped': grouped,
                    'grouped_2': grouped_2,
                    'est_positions': est_positions
                }
                self.all_results.append(iteration_result)
                
                print("  ✅ Processing completed successfully")
                if self.iteration_count == 37:
                    from target_tracking import plot_data
                    plot_data(grouped)

            except Exception as e:
                print(f"  ❌ Error during processing: {e}")
                # Reset current state on error
                self.current_tracked_devices = []
                self.current_estimated_positions = []
                self.current_grouped_data = pd.DataFrame()
        else:
            print("  ⚠️  No data to process")
            # Reset current state when no data
            self.current_tracked_devices = []
            self.current_estimated_positions = []
            self.current_grouped_data = pd.DataFrame()
        
        # Move to next time window
        self.current_time = window_end
    
    def prepare_data_for_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare database data for main processing function
        
        Args:
            data: raw data from database
            
        Returns:
            processed data ready for main function
        """
        if len(data) == 0:
            return data
        
        # Convert created_at to relative seconds from start
        if 'created_at' in data.columns:
            if not pd.api.types.is_datetime64_any_dtype(data['created_at']):
                data['created_at'] = pd.to_datetime(data['created_at'])
            
            # Create relative time in seconds
            data['time'] = (data['created_at'] - self.start_time).dt.total_seconds()
        
        return data
    
    def save_iteration_results(self):
        """Save current iteration results to Excel"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        iteration_str = f"iter_{self.iteration_count:03d}"
        
        # Create output directory
        output_dir = "database_bluetooth_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save estimated positions if available
        if len(self.current_estimated_positions) > 0:
            positions_filename = f"{iteration_str}_positions_{timestamp}.xlsx"
            positions_path = os.path.join(output_dir, positions_filename)
            
            positions_df = pd.DataFrame(self.current_estimated_positions)
            positions_df.to_excel(positions_path, index=False, sheet_name='Positions')
            print(f"    💾 Positions saved: {positions_filename}")
        
        # Save grouped data if available
        if len(self.current_grouped_data) > 0:
            grouped_filename = f"{iteration_str}_grouped_{timestamp}.xlsx"
            grouped_path = os.path.join(output_dir, grouped_filename)
            
            self.current_grouped_data.to_excel(grouped_path, index=False, sheet_name='Grouped_Data')
            print(f"    💾 Grouped data saved: {grouped_filename}")
        
        # Save current state summary
        self.save_current_state_summary(output_dir, iteration_str, timestamp)
    
    def save_current_state_summary(self, output_dir: str, iteration_str: str, timestamp: str):
        """Save summary of current state"""
        state_summary = {
            'iteration': [self.iteration_count],
            'timestamp': [self.current_time.strftime('%Y-%m-%d %H:%M:%S')],
            'tracked_devices_count': [len(self.current_tracked_devices)],
            'position_estimates_count': [len(self.current_estimated_positions)],
            'grouped_data_records': [len(self.current_grouped_data)],
            'accumulated_records': [len(self.accumulated_data)]
        }
        
        state_df = pd.DataFrame(state_summary)
        state_filename = f"{iteration_str}_state_{timestamp}.xlsx"
        state_path = os.path.join(output_dir, state_filename)
        state_df.to_excel(state_path, index=False, sheet_name='Current_State')
    
    def get_current_state(self) -> Dict:
        """
        Get current state of tracked devices and positions
        
        Returns:
            Dictionary with current state information
        """
        return {
            'iteration': self.iteration_count,
            'current_time': self.current_time,
            'tracked_devices': self.current_tracked_devices,
            'estimated_positions': self.current_estimated_positions,
            'grouped_data': self.current_grouped_data,
            'accumulated_records': len(self.accumulated_data)
        }
    
    def run_simulation(self, pause_duration: float = 2.0):
        """
        Run the complete database simulation
        
        Args:
            pause_duration: seconds to pause between iterations
        """
        print(f"\n🚀 Starting database bluetooth simulation...")
        print(f"Pause between iterations: {pause_duration} seconds")
        print("=" * 60)
        
        start_timestamp = time.time()
        
        while self.current_time < self.end_time:
            # Process current iteration
            self.process_current_iteration()
            
            # Print current state for monitoring
            state = self.get_current_state()
            print(f"  📊 Current state: {len(state['tracked_devices'])} devices, {len(state['estimated_positions'])} positions")
            
            # Pause between iterations
            if pause_duration > 0:
                print(f"  ⏳ Waiting {pause_duration} seconds...")
                time.sleep(pause_duration)
        
        # Final summary
        end_timestamp = time.time()
        total_duration = end_timestamp - start_timestamp
        
        print("\n" + "=" * 60)
        print("🏁 Database simulation completed!")
        print(f"Total iterations: {self.iteration_count}")
        print(f"Total duration: {total_duration:.1f} seconds")
        print(f"Time range processed: {self.start_time} - {self.end_time}")
        
        # Final state
        final_state = self.get_current_state()
        print(f"Final tracked devices: {len(final_state['tracked_devices'])}")
        print(f"Final position estimates: {len(final_state['estimated_positions'])}")

def cleanup_files():
    """
    Clean up files in current directory:
    - Deletes 'metadata.json' completely if it exists
    - Empties 'output.csv' (file remains but becomes empty)
    """
    try:
        # Delete metadata.json completely
        if os.path.exists("metadata.json"):
            os.remove("metadata.json")
            print("✓ Deleted: metadata.json")
        else:
            print("ℹ File not found: metadata.json (nothing to delete)")
        
        # Reset output.csv with header row
        with open("output.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Classification", "Latitude", "Longitude", "Fingerprint"])
        print("✓ Reset: output.csv (with headers)")
        
        print("🧹 File cleanup completed successfully!")
        
    except PermissionError as e:
        print(f"❌ Permission denied: {e}")
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

def main():
    """Main function to run the database simulation"""
    
    # Database configuration
    db_config = {
        'database': 'rasp',
        'user': 'simje951',
        'password': 'xA9f8E7G6emt',
        'host': '192.168.1.10',
        'port': '5432'
    }
    
    # Time configuration - moved to main
    start_time = datetime(2025, 8, 27, 17, 49, 28)  # Starting time 2025-08-21 16:21:48
    end_time = datetime(2025, 8, 27, 18, 8, 0)     # Ending time
    
    cleanup_files()
    
    # Simulation configuration
    update_interval = 30  # seconds
    lookback_window = 180  # seconds
    pause_duration = 1   # seconds between iterations
    
    try:
        # Create database controller
        controller = DatabaseBluetoothController(
            db_config=db_config,
            start_time=start_time,
            end_time=end_time,
            update_interval_seconds=update_interval,
            lookback_window_seconds=lookback_window
        )
        
        # Run simulation
        controller.run_simulation(pause_duration=pause_duration)
        
        # Access final state variables
        print(f"\n📋 Final state variables:")
        print(f"current_tracked_devices: {len(controller.current_tracked_devices)} devices")
        print(f"current_estimated_positions: {len(controller.current_estimated_positions)} positions")
        print(f"current_grouped_data shape: {controller.current_grouped_data.shape}")
        
        # Example of accessing the variables
        tracked_devices = controller.current_tracked_devices
        estimated_positions = controller.current_estimated_positions
        
        print("\n🎯 You now have access to these variables:")
        print(f"tracked_devices = {tracked_devices}")
        print(f"estimated_positions = {estimated_positions}")
        
    except Exception as e:
        print(f"❌ Simulation error: {e}")

if __name__ == "__main__":
    main()