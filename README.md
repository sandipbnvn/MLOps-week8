# Data Poisoning Experiments in Machine Learning Models

**Author:** Sandip Biswas  
**Roll No:** 21F1002787  
**Course:** MLOps Course - IITM BS Degree  
**Week:** 8  
**Project:** Data Poisoning Experiments with Version Control and MLflow Tracking

## Overview

This project demonstrates the impact of data poisoning attacks on machine learning models, specifically focusing on image classification tasks. The experiments investigate how intentional data corruption affects model performance and training dynamics. The project uses DVC (Data Version Control) for data management, MLflow for experiment tracking, and implements a systematic approach to study poisoning effects.

## Dataset and Model Architecture

### Dataset
- **Source:** Cat and Dog classification dataset from Kaggle
- **Structure:** 
  - Training data: 2000 images (1000 cats + 1000 dogs)
  - Validation data: 800 images (400 cats + 400 dogs)
  - Image dimensions: 150x150 pixels
- **Directory Structure:**
  ```
  data/
  ├── train/
  │   ├── cats/ (1000 images)
  │   └── dogs/ (1000 images)
  └── validation/
      ├── cats/ (400 images)
      └── dogs/ (400 images)
  ```

### Model Architecture
- **Base Model:** VGG16 (pre-trained on ImageNet)
- **Feature Extraction:** Transfer learning using bottleneck features
- **Top Model:** 
  - Flatten layer
  - Dense layer (256 units, ReLU activation)
  - Dropout (0.5)
  - Output layer (1 unit, sigmoid activation)
- **Training Parameters:**
  - Epochs: 10
  - Batch size: 10
  - Optimizer: RMSprop
  - Loss function: Binary crossentropy
  - Metrics: Accuracy

## Data Poisoning Methodology

### Poisoning Strategy
The poisoning attack involves swapping images between classes to create mislabeled training data:

1. **Random Selection:** Images are randomly selected from both cat and dog classes
2. **Cross-Class Movement:** 
   - Cat images are moved to the dog folder
   - Dog images are moved to the cat folder
3. **Balanced Poisoning:** Equal number of images are moved from each class to maintain balance

### Poisoning Levels Tested
The experiments were conducted with the following poisoning percentages:
- **0%** (baseline - no poisoning)
- **1%** (20 images swapped)
- **2.5%** (50 images swapped)
- **5%** (100 images swapped)
- **7.5%** (150 images swapped)
- **10%** (200 images swapped)

## Experimental Setup

### Tools and Technologies
- **DVC:** Data version control for tracking dataset changes
- **MLflow:** Experiment tracking and model versioning
- **TensorFlow/Keras:** Deep learning framework
- **Git:** Version control for code and experiment tracking
- **Python Libraries:**
  - `tensorflow>=2,<3`
  - `mlflow`
  - `dvc`
  - `pandas`
  - `matplotlib`
  - `tqdm>=4.41.0,<5`
  - `pillow`
  - `scipy`

### Experiment Workflow
Each poisoning experiment follows this systematic process:

1. **Dataset Restoration:** Restore original dataset state using DVC
2. **Data Poisoning:** Apply specified poisoning percentage using `poison_data.py`
3. **Model Training:** Train the model using `train.py` with MLflow tracking
4. **Metrics Logging:** Record training and validation metrics
5. **Visualization:** Generate performance plots using `plot_metrics.py`
6. **Version Control:** Commit changes to Git with descriptive messages
7. **Artifact Storage:** Store models and metrics in MLflow

## Results and Analysis

### Performance Metrics Tracked
For each experiment, the following metrics were recorded:
- **Training Loss:** Model loss on training data
- **Training Accuracy:** Model accuracy on training data
- **Validation Loss:** Model loss on validation data
- **Validation Accuracy:** Model accuracy on validation data

### Key Findings

#### Baseline Performance (0% Poisoning)
- Final Training Accuracy: ~94.25%
- Final Validation Accuracy: ~84.75%
- Training Loss: ~0.146
- Validation Loss: ~0.549

#### Impact of Poisoning
1. **Training Accuracy:** Generally increases with poisoning (overfitting effect)
2. **Validation Accuracy:** Decreases with higher poisoning levels
3. **Generalization Gap:** Widens as poisoning increases
4. **Model Robustness:** Degrades significantly with >5% poisoning

### Visual Results
The project includes performance plots for each poisoning level:
- `poison_0_percent_metrics.jpg` - Baseline performance
- `poison_1_percent_metrics.jpg` - 1% poisoning effects
- `poison_2.5_percent_metrics.jpg` - 2.5% poisoning effects
- `poison_5_percent_metrics.jpg` - 5% poisoning effects
- `poison_7.5_percent_metrics.jpg` - 7.5% poisoning effects
- `poison_10_percent_metrics.jpg` - 10% poisoning effects

## Experiment Execution

### Running Individual Experiments
```bash
# Run single poisoning experiment
python run_experiment.py --p 5.0

# Start MLflow server and run experiments
python run_experiment.py --start-mlflow
```

### Running All Experiments
```bash
# Run all poisoning levels (0%, 1%, 2.5%, 5%, 7.5%, 10%)
python run_experiment.py
```

### Manual Execution Steps
1. **Poison the dataset:**
   ```bash
   python poison_data.py --p <poisoning_percentage>
   ```

2. **Train the model:**
   ```bash
   python train.py --p <poisoning_percentage>
   ```

3. **Generate plots:**
   ```bash
   python plot_metrics.py --p <poisoning_percentage> --output <filename>
   ```

## Version Control and Reproducibility

### DVC Configuration
- **Data tracking:** `data.dvc` tracks the dataset
- **Model tracking:** `model.weights.h5.dvc` tracks trained models
- **Cache management:** Local DVC cache for efficient storage

### Git Workflow
- Each experiment is committed with descriptive messages
- Tags are used to mark significant model versions
- Branch strategy: `main` branch for stable experiments

### MLflow Integration
- **Tracking URI:** `http://localhost:5000`
- **Parameter logging:** Poisoning percentage, epochs, batch size
- **Metric tracking:** Training and validation metrics per epoch
- **Model storage:** TensorFlow models with metadata
- **Artifact logging:** Metrics CSV files and plots

## Security Implications

### Attack Vector Analysis
- **Data Integrity:** Poisoning affects training data quality
- **Model Trust:** Compromised models may make incorrect predictions
- **Detection Challenges:** Poisoning can be subtle and hard to detect

### Mitigation Strategies
1. **Data Validation:** Implement robust data validation pipelines
2. **Model Monitoring:** Continuous monitoring of model performance
3. **Version Control:** Track all data and model changes
4. **Reproducibility:** Ensure experiments can be reproduced

## Technical Implementation Details

### File Structure
```
example-versioning/
├── data/                    # Dataset directory
├── mlruns/                  # MLflow experiment tracking
├── mlartifacts/            # MLflow model artifacts
├── train.py                # Main training script
├── poison_data.py          # Data poisoning implementation
├── run_experiment.py       # Experiment orchestration
├── plot_metrics.py         # Visualization script
├── requirements.txt        # Python dependencies
├── data.dvc               # DVC data tracking
├── model.weights.h5.dvc   # DVC model tracking
├── commands.txt           # Execution commands
└── *.jpg                  # Performance plots
```

### Key Scripts

#### `poison_data.py`
- Implements the poisoning algorithm
- Randomly selects and moves images between classes
- Maintains balanced poisoning across classes

#### `train.py`
- Implements the VGG16-based transfer learning model
- Integrates MLflow for experiment tracking
- Handles Unicode issues on Windows systems

#### `run_experiment.py`
- Orchestrates the complete experiment workflow
- Manages dataset restoration between experiments
- Handles MLflow server startup and configuration

#### `plot_metrics.py`
- Generates performance visualization plots
- Creates side-by-side loss and accuracy plots
- Saves high-resolution images for analysis

## Conclusion

This project successfully demonstrates the impact of data poisoning attacks on machine learning models. The experiments show that even small amounts of poisoned data can significantly affect model performance and generalization. The implementation provides a robust framework for studying data poisoning effects with proper version control and experiment tracking.

### Key Takeaways
1. **Data Quality Matters:** Even 1% poisoning can impact model performance
2. **Monitoring is Critical:** Continuous monitoring helps detect poisoning
3. **Version Control is Essential:** Tracking changes enables reproducibility
4. **Security Awareness:** ML systems are vulnerable to data poisoning attacks

### Future Work
- Implement poisoning detection algorithms
- Study defense mechanisms against poisoning attacks
- Extend experiments to other datasets and model architectures
- Develop automated poisoning detection pipelines

---

**Note:** This project is part of the MLOps course curriculum at IITM BS Degree, focusing on practical aspects of machine learning operations, security, and reproducibility.
