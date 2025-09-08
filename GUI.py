import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import psycopg2
from typing import Dict, Optional

# Import your existing classes
from main import process_bluetooth_data

class BluetoothGUIController:
    def __init__(self, root):
        self.root = root
        self.root.title("Bluetooth Database Controller")
        self.root.geometry("1200x800")
        
        # Simulation state
        self.is_running = False
        self.controller = None
        self.simulation_thread = None
        
        # Data for plotting
        self.iteration_data = []
        self.device_counts = []
        self.estimate_counts = []
        self.timestamps = []
        
        # Database configuration
        self.db_config = {
            'database': 'rasp',
            'user': 'simje951',
            'password': 'xA9f8E7G6emt',
            'host': '192.168.1.10',
            'port': '5432'
        }
        
        self.setup_gui()
        self.setup_plots()
        
    def setup_gui(self):
        """Setup the GUI components"""
        # Get current time for defaults
        current_time = datetime.now()
        end_time = current_time + timedelta(hours=2)
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control Panel
        control_frame = ttk.LabelFrame(main_frame, text="Simulation Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Time Controls
        time_frame = ttk.Frame(control_frame)
        time_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Start Time - set to current time
        ttk.Label(time_frame, text="Start Time:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.start_date_var = tk.StringVar(value=current_time.strftime("%Y-%m-%d"))
        self.start_time_var = tk.StringVar(value=current_time.strftime("%H:%M:%S"))
        ttk.Entry(time_frame, textvariable=self.start_date_var, width=12).grid(row=0, column=1, padx=(0, 5))
        ttk.Entry(time_frame, textvariable=self.start_time_var, width=10).grid(row=0, column=2, padx=(0, 20))
        
        # End Time - set to current time + 2 hours
        ttk.Label(time_frame, text="End Time:").grid(row=0, column=3, sticky=tk.W, padx=(0, 5))
        self.end_date_var = tk.StringVar(value=end_time.strftime("%Y-%m-%d"))
        self.end_time_var = tk.StringVar(value=end_time.strftime("%H:%M:%S"))
        ttk.Entry(time_frame, textvariable=self.end_date_var, width=12).grid(row=0, column=4, padx=(0, 5))
        ttk.Entry(time_frame, textvariable=self.end_time_var, width=10).grid(row=0, column=5)
        
        # Data Mode Selection
        mode_frame = ttk.LabelFrame(control_frame, text="Data Collection Mode", padding="5")
        mode_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Set default to cumulative
        self.data_mode_var = tk.StringVar(value="cumulative")
        
        ttk.Radiobutton(mode_frame, text="Lookback Window", variable=self.data_mode_var, 
                       value="lookback", command=self.on_mode_change).grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        ttk.Radiobutton(mode_frame, text="Cumulative from Start", variable=self.data_mode_var, 
                       value="cumulative", command=self.on_mode_change).grid(row=0, column=1, sticky=tk.W)
        
        # Mode description - set initial description for cumulative mode
        self.mode_desc_var = tk.StringVar(value="Accumulates data from start time with increasing windows")
        ttk.Label(mode_frame, textvariable=self.mode_desc_var, font=("TkDefaultFont", 8), 
                 foreground="gray").grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # Parameter Controls
        param_frame = ttk.Frame(control_frame)
        param_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Update Interval
        ttk.Label(param_frame, text="Update Interval (s):").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.update_interval_var = tk.StringVar(value="30")
        ttk.Entry(param_frame, textvariable=self.update_interval_var, width=8).grid(row=0, column=1, padx=(0, 20))
        
        # Lookback Window (conditional) - hidden by default since cumulative is selected
        self.lookback_label = ttk.Label(param_frame, text="Lookback Window (s):")
        self.lookback_var = tk.StringVar(value="60")
        self.lookback_entry = ttk.Entry(param_frame, textvariable=self.lookback_var, width=8)
        # Don't grid these initially since cumulative mode is default
        
        # Pause Duration
        ttk.Label(param_frame, text="Pause Duration (s):").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        self.pause_var = tk.StringVar(value="30")
        ttk.Entry(param_frame, textvariable=self.pause_var, width=8).grid(row=0, column=5)
        
        # Control Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=(10, 0))
        
        self.play_button = ttk.Button(button_frame, text="▶ Start", command=self.start_simulation)
        self.play_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="⏹ Stop", command=self.stop_simulation, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.reset_button = ttk.Button(button_frame, text="🔄 Reset", command=self.reset_data)
        self.reset_button.pack(side=tk.LEFT)
        
        # Status Panel
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        status_frame.columnconfigure(0, weight=1)
        
        # Status Labels
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var, font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Statistics
        stats_frame = ttk.Frame(status_frame)
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(stats_frame, text="Current Iteration:", font=("TkDefaultFont", 9)).grid(row=0, column=0, sticky=tk.W)
        self.iteration_var = tk.StringVar(value="0")
        ttk.Label(stats_frame, textvariable=self.iteration_var, font=("TkDefaultFont", 9, "bold")).grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(stats_frame, text="Tracked Devices:", font=("TkDefaultFont", 9)).grid(row=1, column=0, sticky=tk.W)
        self.devices_var = tk.StringVar(value="0")
        ttk.Label(stats_frame, textvariable=self.devices_var, font=("TkDefaultFont", 9, "bold")).grid(row=1, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(stats_frame, text="Position Estimates:", font=("TkDefaultFont", 9)).grid(row=2, column=0, sticky=tk.W)
        self.estimates_var = tk.StringVar(value="0")
        ttk.Label(stats_frame, textvariable=self.estimates_var, font=("TkDefaultFont", 9, "bold")).grid(row=2, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(stats_frame, text="Accumulated Records:", font=("TkDefaultFont", 9)).grid(row=3, column=0, sticky=tk.W)
        self.records_var = tk.StringVar(value="0")
        ttk.Label(stats_frame, textvariable=self.records_var, font=("TkDefaultFont", 9, "bold")).grid(row=3, column=1, sticky=tk.W, padx=(5, 0))
        
        # Current Time Window
        ttk.Label(stats_frame, text="Current Window:", font=("TkDefaultFont", 9)).grid(row=4, column=0, sticky=tk.W, pady=(10, 0))
        self.current_window_var = tk.StringVar(value="--:--:-- to --:--:--")
        ttk.Label(stats_frame, textvariable=self.current_window_var, font=("TkDefaultFont", 8, "bold")).grid(row=4, column=1, sticky=tk.W, padx=(5, 0), pady=(10, 0))
        
        # Log area
        log_frame = ttk.LabelFrame(status_frame, text="Log", padding="5")
        log_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = tk.Text(log_frame, height=8, width=40, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
    def on_mode_change(self):
        """Handle data mode selection change"""
        mode = self.data_mode_var.get()
        if mode == "lookback":
            self.mode_desc_var.set("Uses a sliding window of fixed duration")
            self.lookback_label.grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
            self.lookback_entry.grid(row=0, column=3, padx=(0, 20))
        else:  # cumulative
            self.mode_desc_var.set("Accumulates data from start time with increasing windows")
            self.lookback_label.grid_remove()
            self.lookback_entry.grid_remove()
        
    def setup_plots(self):
        """Setup the matplotlib plots"""
        # Plot Frame - Fixed reference to main_frame
        plot_frame = ttk.LabelFrame(list(self.root.children.values())[0], text="Real-time Data", padding="5")
        plot_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.fig.tight_layout(pad=3.0)
        
        # Setup plots
        self.ax1.set_title("Tracked Devices Over Time")
        self.ax1.set_ylabel("Number of Devices")
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title("Position Estimates Over Time")
        self.ax2.set_ylabel("Number of Estimates")
        self.ax2.set_xlabel("Iteration")
        self.ax2.grid(True, alpha=0.3)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initialize empty plots
        self.device_line, = self.ax1.plot([], [], 'b-o', markersize=4, linewidth=2)
        self.estimate_line, = self.ax2.plot([], [], 'r-o', markersize=4, linewidth=2)
        
    def log_message(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.root.after(0, lambda: self._add_to_log(formatted_message))
    
    def _add_to_log(self, message):
        """Thread-safe method to add to log"""
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        
        # Limit log size
        lines = int(self.log_text.index('end-1c').split('.')[0])
        if lines > 100:
            self.log_text.delete('1.0', '50.0')
    
    def update_stats(self, iteration, devices, estimates, records, window_start, window_end):
        """Update statistics display"""
        self.root.after(0, lambda: self._update_stats_ui(iteration, devices, estimates, records, window_start, window_end))
    
    def _update_stats_ui(self, iteration, devices, estimates, records, window_start, window_end):
        """Thread-safe method to update stats"""
        self.iteration_var.set(str(iteration))
        self.devices_var.set(str(devices))
        self.estimates_var.set(str(estimates))
        self.records_var.set(str(records))
        self.current_window_var.set(f"{window_start.strftime('%H:%M:%S')} to {window_end.strftime('%H:%M:%S')}")
    
    def update_plots(self, devices, estimates):
        """Update the plots with new data"""
        self.device_counts.append(devices)
        self.estimate_counts.append(estimates)
        
        iterations = list(range(1, len(self.device_counts) + 1))
        
        self.root.after(0, lambda: self._update_plots_ui(iterations, self.device_counts, self.estimate_counts))
    
    def _update_plots_ui(self, iterations, device_counts, estimate_counts):
        """Thread-safe method to update plots"""
        # Update device plot
        self.device_line.set_data(iterations, device_counts)
        self.ax1.relim()
        self.ax1.autoscale_view()
        
        # Update estimate plot  
        self.estimate_line.set_data(iterations, estimate_counts)
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # Refresh canvas
        self.canvas.draw()
    
    def parse_datetime(self, date_str, time_str):
        """Parse date and time strings into datetime object"""
        try:
            datetime_str = f"{date_str} {time_str}"
            return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            raise ValueError(f"Invalid date/time format: {e}")
    
    def start_simulation(self):
        """Start the bluetooth simulation"""
        if self.is_running:
            return
            
        try:
            # Parse parameters
            start_time = self.parse_datetime(self.start_date_var.get(), self.start_time_var.get())
            end_time = self.parse_datetime(self.end_date_var.get(), self.end_time_var.get())
            update_interval = int(self.update_interval_var.get())
            pause_duration = float(self.pause_var.get())
            data_mode = self.data_mode_var.get()
            
            # Get lookback window only if in lookback mode
            lookback_window = None
            if data_mode == "lookback":
                lookback_window = int(self.lookback_var.get())
            
            # Validation
            if end_time <= start_time:
                messagebox.showerror("Error", "End time must be after start time")
                return
                
            if update_interval <= 0:
                messagebox.showerror("Error", "Update interval must be positive")
                return
                
            if data_mode == "lookback" and lookback_window <= 0:
                messagebox.showerror("Error", "Lookback window must be positive")
                return
            
            # Update UI state
            self.is_running = True
            self.play_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_var.set("Starting simulation...")
            
            # Reset data
            self.device_counts.clear()
            self.estimate_counts.clear()
            self.timestamps.clear()
            
            # Start simulation in separate thread
            self.simulation_thread = threading.Thread(
                target=self.run_simulation,
                args=(start_time, end_time, update_interval, lookback_window, pause_duration, data_mode)
            )
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            
            mode_text = "lookback window" if data_mode == "lookback" else "cumulative from start"
            self.log_message(f"Simulation started in {mode_text} mode")
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameters: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start simulation: {e}")
    
    def stop_simulation(self):
        """Stop the bluetooth simulation"""
        self.is_running = False
        self.play_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Stopping...")
        self.log_message("Stop requested")
    
    def reset_data(self):
        """Reset all data and plots"""
        if self.is_running:
            self.stop_simulation()
        
        self.device_counts.clear()
        self.estimate_counts.clear()
        self.timestamps.clear()
        
        # Clear plots
        self.device_line.set_data([], [])
        self.estimate_line.set_data([], [])
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.canvas.draw()
        
        # Reset stats
        self.iteration_var.set("0")
        self.devices_var.set("0")
        self.estimates_var.set("0")
        self.records_var.set("0")
        self.current_window_var.set("--:--:-- to --:--:--")
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        
        self.status_var.set("Reset complete")
        self.log_message("Data reset")
    
    def run_simulation(self, start_time, end_time, update_interval, lookback_window, pause_duration, data_mode):
        """Run the actual simulation (called in separate thread)"""
        try:
            from external_time_controller_db import DatabaseBluetoothController, cleanup_files
            
            self.log_message("Initializing database controller...")
            
            # Cleanup files
            cleanup_files()
            
            # Create controller - use a default lookback for initialization
            controller = DatabaseBluetoothController(
                db_config=self.db_config,
                start_time=start_time,
                end_time=end_time,
                update_interval_seconds=update_interval,
                lookback_window_seconds=lookback_window if lookback_window else update_interval
            )
            
            mode_text = "lookback window" if data_mode == "lookback" else "cumulative from start"
            self.root.after(0, lambda: self.status_var.set(f"Simulation running ({mode_text})"))
            
            current_time = start_time
            iteration = 0
            
            while current_time < end_time and self.is_running:
                iteration += 1
                
                if data_mode == "lookback":
                    # Original lookback window mode
                    window_start = current_time
                    window_end = current_time + timedelta(seconds=update_interval)
                    
                    # Ensure we don't exceed end time
                    if window_end > end_time:
                        window_end = end_time
                else:
                    # Cumulative mode - always start from beginning, extend window
                    window_start = start_time
                    window_end = start_time + timedelta(seconds=update_interval * iteration)
                    
                    # Ensure we don't exceed end time
                    if window_end > end_time:
                        window_end = end_time
                
                self.log_message(f"Iteration {iteration}: {window_start.strftime('%H:%M:%S')} - {window_end.strftime('%H:%M:%S')}")
                
                try:
                    # Query database for the window
                    new_data = controller.query_database_window(window_start, window_end)
                    
                    if data_mode == "lookback":
                        # In lookback mode, update accumulated data normally
                        controller.update_accumulated_data(new_data)
                    else:
                        # In cumulative mode, replace accumulated data with full window data
                        controller.accumulated_data = new_data
                    
                    records_count = len(controller.accumulated_data)
                    devices_count = 0
                    estimates_count = 0
                    
                    # Process data if available
                    if records_count > 0:
                        processed_data = controller.prepare_data_for_processing(controller.accumulated_data.copy())
                        
                        grouped, grouped_2, tracked_devices, est_positions, final_data = process_bluetooth_data(
                            processed_data,
                            time_window=5,
                            num_of_anchors=3,
                            position=1
                        )
                        
                        devices_count = len(tracked_devices)
                        estimates_count = len(est_positions)
                        
                        self.log_message(f"Processed: {devices_count} devices, {estimates_count} estimates")
                    
                    # Update UI
                    self.update_stats(iteration, devices_count, estimates_count, records_count, window_start, window_end)
                    self.update_plots(devices_count, estimates_count)
                    
                except Exception as e:
                    self.log_message(f"Processing error: {e}")
                
                # Move to next iteration
                if data_mode == "lookback":
                    current_time = window_end
                else:
                    # In cumulative mode, check if we've reached the end
                    if window_end >= end_time:
                        break
                    current_time = start_time + timedelta(seconds=update_interval * iteration)
                
                # Pause if simulation is still running
                if self.is_running and pause_duration > 0:
                    time.sleep(pause_duration)
            
            self.root.after(0, lambda: self.status_var.set("Simulation completed"))
            self.log_message("Simulation completed")
            
        except Exception as e:
            self.log_message(f"Simulation error: {e}")
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
        finally:
            self.root.after(0, lambda: self.stop_simulation())

def main():
    root = tk.Tk()
    app = BluetoothGUIController(root)
    root.mainloop()

if __name__ == "__main__":
    main()