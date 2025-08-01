import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_metrics(csv_file, p_value, output_file):
    """
    Plot training and validation metrics from CSV file.
    
    Args:
        csv_file (str): Path to metrics CSV file
        p_value (float): Poisoning percentage for filename
        output_file (str): Output filename for the plot
    """
    # Read metrics from CSV
    df = pd.read_csv(csv_file)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    ax1.plot(df['epoch'], df['loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(df['epoch'], df['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss vs Epoch (Poisoning: {p_value}%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(df['epoch'], df['accuracy'] * 100, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(df['epoch'], df['val_accuracy'] * 100, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'Accuracy vs Epoch (Poisoning: {p_value}%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--csv', type=str, default='metrics.csv', help='Metrics CSV file')
    parser.add_argument('--p', type=float, required=True, help='Poisoning percentage')
    parser.add_argument('--output', type=str, help='Output filename')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f"poison_{args.p}_percent_metrics.jpg"
    
    plot_metrics(args.csv, args.p, args.output) 