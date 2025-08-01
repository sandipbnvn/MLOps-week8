#!/usr/bin/env python3
"""
Simple Poisoning Experiment Runner

This script runs poisoning experiments without the Unicode issues on Windows.
"""

import os
import subprocess
import sys
import argparse

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*50)
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True, encoding='utf-8')
    
    if result.returncode != 0:
        print(f"Error running command: {command}")
        print(f"Error: {result.stderr}")
        return False
    
    print(f"Success: {result.stdout}")
    return True

def run_single_experiment(p_value):
    """Run a single poisoning experiment."""
    print(f"\n{'#'*60}")
    print(f"Starting experiment with poisoning level: {p_value}%")
    print(f"{'#'*60}")
    
    # 1. Poison the dataset
    if not run_command(f"python poison_data.py --p {p_value}", f"Poisoning dataset with {p_value}%"):
        return False
    
    # 2. Train the model
    if not run_command(f"python train.py --p {p_value}", f"Training model with {p_value}% poisoning"):
        return False
    
    # 3. Generate plot
    plot_filename = f"poison_{p_value}_percent_metrics.jpg"
    if not run_command(f"python plot_metrics.py --p {p_value} --output {plot_filename}", 
                      f"Generating plot for {p_value}% poisoning"):
        return False
    
    # 4. Add files to git
    if not run_command("git add .", "Adding all files to git"):
        return False
    
    # 5. Commit
    commit_message = f"Poisoning experiment: {p_value}% data corruption"
    if not run_command(f'git commit -m "{commit_message}"', f"Committing {p_value}% experiment"):
        return False
    
    print(f"\n✅ Experiment completed for {p_value}% poisoning")
    return True

def restore_dataset():
    """Restore the original dataset state from DVC."""
    print("Restoring original dataset state...")
    
    result = subprocess.run("dvc checkout data.dvc", shell=True, capture_output=True, text=True, encoding='utf-8')
    
    if result.returncode != 0:
        print(f"Error restoring dataset: {result.stderr}")
        return False
    
    print("✅ Dataset restored to original state")
    return True

def main():
    parser = argparse.ArgumentParser(description='Run poisoning experiments')
    parser.add_argument('--p', type=float, help='Single poisoning percentage to test')
    parser.add_argument('--p-values', nargs='+', type=float, 
                       default=[0, 1, 2.5, 5, 7.5, 10],
                       help='List of poisoning percentages to test')
    parser.add_argument('--start-mlflow', action='store_true',
                       help='Start MLflow server before experiments')
    
    args = parser.parse_args()
    
    # If single p value is provided, use that
    if args.p is not None:
        p_values = [args.p]
    else:
        p_values = args.p_values
    
    # Start MLflow server if requested
    if args.start_mlflow:
        print("Starting MLflow server...")
        mlflow_process = subprocess.Popen(
            "mlflow server --host 0.0.0.0 --port 5000",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("MLflow server started on http://localhost:5000")
    
    # Set MLflow tracking URI
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    
    # Run experiments for each poisoning level
    for i, p in enumerate(p_values):
        try:
            # Restore dataset to original state (except for first experiment)
            if i > 0:
                if not restore_dataset():
                    print(f"Failed to restore dataset, skipping experiment {p}%")
                    continue
            
            if not run_single_experiment(p):
                print(f"Experiment {p}% failed, continuing with next...")
                continue
                
        except KeyboardInterrupt:
            print("\nExperiment interrupted by user")
            break
        except Exception as e:
            print(f"Error in experiment {p}%: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    if args.start_mlflow:
        print(f"MLflow UI available at: http://localhost:5000")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 