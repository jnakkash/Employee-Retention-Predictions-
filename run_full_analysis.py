#!/usr/bin/env python3
"""
Full Analysis Pipeline for Salifort Motors Employee Retention Project
This script runs the complete analysis pipeline in sequence:
1. Data preprocessing and exploratory analysis
2. Model training
3. Model evaluation
4. Dashboard creation
"""

import os
import subprocess
import time
from pathlib import Path

def print_section(title):
    """Print a section title with formatting."""
    border = "=" * (len(title) + 10)
    print(f"\n{border}")
    print(f"===  {title}  ===")
    print(f"{border}\n")

def run_script(script_name, description):
    """Run a Python script and capture its output."""
    print_section(description)
    try:
        print(f"Starting {script_name}...")
        result = subprocess.run(['python', script_name], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings/Errors from {script_name}:")
            print(result.stderr)
        print(f"{script_name} completed successfully.\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e.stderr)
        return False

def create_directory_structure():
    """Create the project directory structure if it doesn't exist."""
    directories = [
        "../data/raw",
        "../data/processed",
        "../results/analyzed_data",
        "../results/model_comparison",
        "../models",
        "../documentation"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    """Run the full analysis pipeline."""
    start_time = time.time()
    
    print_section("SALIFORT MOTORS EMPLOYEE RETENTION ANALYSIS")
    print("Starting full analysis pipeline...")
    
    # Create directory structure
    print_section("CREATING DIRECTORY STRUCTURE")
    create_directory_structure()
    
    # Check if input data exists
    data_path = "./data/raw/HR_capstone_dataset.csv"
    if not os.path.exists(data_path):
        print("Error: Input dataset not found.")
        print(f"Please ensure the dataset HR_capstone_dataset.csv exists at {data_path} before running this script.")
        return
    
    # Run exploratory analysis
    if not run_script("exploratory_analysis.py", "EXPLORATORY DATA ANALYSIS"):
        print("Exploratory analysis failed. Stopping pipeline.")
        return
    
    # Run model training
    if not run_script("model_training.py", "MODEL TRAINING"):
        print("Model training failed. Stopping pipeline.")
        return
    
    # Run model evaluation
    if not run_script("model_evaluation.py", "MODEL EVALUATION"):
        print("Model evaluation failed. Stopping pipeline.")
        return
    
    # Create interactive dashboard
    if not run_script("interactive_dashboard.py", "INTERACTIVE DASHBOARD CREATION"):
        print("Dashboard creation failed. Pipeline completed with warnings.")
    
    # Calculate and print total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    minutes, seconds = divmod(execution_time, 60)
    hours, minutes = divmod(minutes, 60)
    
    print_section("ANALYSIS PIPELINE COMPLETED")
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("\nResults available in:")
    print("- Exploratory analysis: ../results/analyzed_data/")
    print("- Model evaluation: ../results/model_comparison/")
    print("- Executive summary: ../documentation/executive_summary.md")
    print("- Interactive dashboard: interactive_dashboard.html")
    print("\nTo view the dashboard, open the interactive_dashboard.html file in your web browser.")

if __name__ == "__main__":
    main() 