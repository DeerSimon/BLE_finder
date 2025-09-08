import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

# Import your main processing script
from main import process_bluetooth_data, save_results_to_files

class BluetoothTimeController:
    """
    External controller for time-based bluetooth data processing
    """
    
    def __init__(self, data_filepath, update_interval=30, lookback_window=60):
        """
        Initialize the time controller
        
        Args:
            data_filepath: path to the data file
            update_interval: interval between updates in seconds
            lookback_window: how far back to keep data in seconds
        """
        self.data_filepath = data_filepath
        self.update_interval = update_interval
        self.lookback_window = lookback_window
        
        # Load and prepare data
        print(f"Loading data from: {data_filepath}")
        self.full_data = self._load_data()
        
        # Time tracking
        self.start_time = self.full_data['Time'].min()
        self.end_time = self.full_data['Time'].max()
        self.current_time = self.start_time
        
        # Accumulated data for processing
        self.accumulated_data = pd.DataFrame()
        
        # Results tracking
        self.iteration_count = 0
        self.all_results = []
        
        print(f"Data loaded successfully!")
        print(f"Total records: {len(self.full_data)}")
        print(f"Time range: {self.start_time:.1f} - {self.end_time:.1f} seconds")
        print(f"Estimated iterations: {int((self.end_time - self.start_time) / self.update_interval)}")
        print("-" * 60)
    
    def _load_data(self):
        """Load and prepare data from file"""
        if not os.path.exists(self.data_filepath):
            raise FileNotFoundError(f"Data file not found: {self.data_filepath}")
        
        data = pd.read_csv(self.data_filepath)
        
        # Ensure we have time column
        if 'time' in data.columns and 'Time' not in data.columns:
            data = data.rename(columns={'time': 'Time'})
        elif 'Time' not in data.columns:
            raise ValueError("No 'Time' or 'time' column found in data")
        
        # Sort by time
        data = data.sort_values('Time').reset_index(drop=True)
        
        return data
    
    def get_data_window(self, start_time, end_time):
        """Get data for specific time window"""
        mask = (self.full_data['Time'] >= start_time) & (self.full_data['Time'] < end_time)
        return self.full_data[mask].copy()
    
    def update_accumulated_data(self, new_data):
        """Update accumulated data with new data and apply lookback window"""
        if len(new_data) > 0:
            # Add new data
            self.accumulated_data = pd.concat([self.accumulated_data, new_data], ignore_index=True)
            
            # Apply lookback window - keep only recent data
            cutoff_time = self.current_time - self.lookback_window
            self.accumulated_data = self.accumulated_data[
                self.accumulated_data['Time'] >= cutoff_time
            ].reset_index(drop=True)
    
    def process_current_iteration(self):
        """Process current time iteration"""
        self.iteration_count += 1
        window_end = self.current_time + self.update_interval
        
        print(f"\n=== Iteration {self.iteration_count} ===")
        print(f"Time window: {self.current_time:.1f} - {window_end:.1f} seconds")
        
        # Get new data for this window
        new_data = self.get_data_window(self.current_time, window_end)
        print(f"New records: {len(new_data)}")
        
        # Update accumulated data
        self.update_accumulated_data(new_data)
        print(f"Accumulated records: {len(self.accumulated_data)}")
        
        # Process data if we have any
        if len(self.accumulated_data) > 0:
            try:
                # Call main processing function
                grouped, grouped_2, tracked_devices, est_positions, processed_data = process_bluetooth_data(
                    self.accumulated_data,
                    time_window=5,
                    num_of_anchors=4,
                    position=1
                )
                
                print(f"Tracked devices: {len(tracked_devices)}")
                print(f"Grouped data shape: {grouped.shape}")
                print(f"Position estimates: {len(est_positions)}")
                
                # Save results to Excel with iteration info
                self.save_iteration_results(est_positions, grouped_2)
                
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
                
                print("✓ Processing completed successfully")
                
            except Exception as e:
                print(f"✗ Error during processing: {e}")
        else:
            print("No data to process")
        
        # Move to next time window
        self.current_time = window_end
    
    def save_iteration_results(self, est_positions, grouped_2):
        """Save results for current iteration to Excel files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        iteration_str = f"iter_{self.iteration_count:03d}"
        
        # Create output directory if it doesn't exist
        output_dir = "bluetooth_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save estimated positions if available
        if est_positions and len(est_positions) > 0:
            positions_filename = f"{iteration_str}_positions_{timestamp}.xlsx"
            positions_path = os.path.join(output_dir, positions_filename)
            
            # Convert to DataFrame for Excel export
            positions_df = pd.DataFrame(est_positions)
            positions_df.to_excel(positions_path, index=False, sheet_name='Positions')
            print(f"  → Positions saved: {positions_filename}")
        
        # Save grouped data if available
        if len(grouped_2) > 0:
            grouped_filename = f"{iteration_str}_grouped_{timestamp}.xlsx"
            grouped_path = os.path.join(output_dir, grouped_filename)
            
            grouped_2.to_excel(grouped_path, index=False, sheet_name='Grouped_Data')
            print(f"  → Grouped data saved: {grouped_filename}")
        
        # Save summary file with all iterations so far
        self.save_summary_excel(output_dir)
    
    def save_summary_excel(self, output_dir):
        """Save summary of all iterations to Excel"""
        summary_data = []
        for result in self.all_results:
            summary_data.append({
                'Iteration': result['iteration'],
                'Start_Time': result['time_window'][0],
                'End_Time': result['time_window'][1],
                'New_Records': result['new_records'],
                'Accumulated_Records': result['accumulated_records'],
                'Tracked_Devices': result['tracked_devices'],
                'Position_Estimates': result['position_estimates']
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(output_dir, "simulation_summary.xlsx")
            summary_df.to_excel(summary_path, index=False, sheet_name='Summary')
    
    def run_simulation(self, pause_duration=2):
        """
        Run the complete time simulation
        
        Args:
            pause_duration: seconds to pause between iterations
        """
        print(f"\n🚀 Starting bluetooth data simulation...")
        print(f"Pause between iterations: {pause_duration} seconds")
        print("=" * 60)
        
        start_timestamp = time.time()
        
        while self.current_time < self.end_time:
            # Process current iteration
            self.process_current_iteration()
            
            # Pause to allow viewing updates
            if pause_duration > 0:
                print(f"Waiting {pause_duration} seconds...")
                time.sleep(pause_duration)
        
        # Final summary
        end_timestamp = time.time()
        total_duration = end_timestamp - start_timestamp
        
        print("\n" + "=" * 60)
        print("🏁 Simulation completed!")
        print(f"Total iterations: {self.iteration_count}")
        print(f"Total duration: {total_duration:.1f} seconds")
        print(f"Data time range processed: {self.start_time:.1f} - {self.end_time:.1f} seconds")
        
        # Save final summary
        output_dir = "bluetooth_results"
        self.save_summary_excel(output_dir)
        print(f"Results saved in: {output_dir}/")

def main():
    """Main function to run the simulation"""
    
    # Configuration
    data_filepath = "D:\\LIU\\Diploma\\diplom\\code\\search_3\\data_3_cut.csv"  # Kalmarden
    update_interval = 10  # seconds
    lookback_window = 60  # seconds
    pause_duration = 2    # seconds between iterations
    
    try:
        # Create controller
        controller = BluetoothTimeController(
            data_filepath=data_filepath,
            update_interval=update_interval,
            lookback_window=lookback_window
        )
        
        # Run simulation
        controller.run_simulation(pause_duration=pause_duration)
        
    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()