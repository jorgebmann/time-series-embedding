# A Comprehensive Evaluation of Time Series Embedding Methods for Classification Tasks

This repository implements various time series embedding methods and provides a framework for quantitative evaluation of these methods on classification tasks. It allows researchers to compare different embedding techniques based on their theoretical foundations and empirical performance.

## Description
This repository contains code for the quantitative evaluation of different time series embedding methods based on their theoretical foundations and application contexts. Unlike previous work, this study quantitatively compares these methods by testing their performance on various classification tasks across different datasets. The results reveal that embedding methods perform differently depending on the dataset and classification algorithm, emphasizing the need for careful model selection and experimentation. 


## Repository Structure

- `/data`: Scripts for downloading and processing datasets
- `/results`: Results from experiments
- `/utils`: Utility functions for data augmentation and visualization
- `/notebooks`: Jupyter notebooks for experiments
- `/scripts`: Implementation scripts


## Dependencies

The main dependencies for this project include:
- **Python 3.12** (required for compatibility with all packages)
- NumPy
- Pandas  
- TensorFlow
- Scikit-learn
- UMAP
- Matplotlib
- Optuna
- **Specialized packages**: giotto-tda, ts2vg, gudhi, keras-tuner
- And more scientific computing packages

All dependencies are specified in the `requirements.txt` file.

## Installation

### Prerequisites

**Important**: This project requires **Python 3.12** for full compatibility with all packages, especially the topological data analysis libraries (gtda, ts2vg, gudhi).

#### Option 1: Using Conda (Recommended)
```bash
# Install/create Python 3.12 environment with conda
conda create -n timeseries_py312 python=3.12 -y
conda activate timeseries_py312
```

#### Option 2: Using pyenv
```bash
# Install Python 3.12 with pyenv
pyenv install 3.12.11
pyenv local 3.12.11
```

### Setting up the environment

```bash
# Clone the repository
git clone https://github.com/imics-lab/time-series-embedding.git
cd time-series-embedding

# Create virtual environment using Python 3.12
python3.12 -m venv venv
# OR if using conda Python 3.12:
# /path/to/conda/envs/timeseries_py312/bin/python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Verify Python version (should be 3.12.x)
python --version

# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Verify installation (quick test)
python -c "import gtda, ts2vg, gudhi; print('✅ All packages installed successfully!')"

# Run comprehensive environment test
python test_environment.py
```

### Troubleshooting

If you encounter issues:

1. **Python version**: Ensure you're using Python 3.12.x:
   ```bash
   python --version  # Should show Python 3.12.x
   ```

2. **Package conflicts**: If packages fail to install, try:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt --force-reinstall
   ```

3. **Missing system dependencies**: On Ubuntu/Debian, you may need:
   ```bash
   sudo apt-get update
   sudo apt-get install python3.12-dev python3.12-venv build-essential
   ```

### Key Package Notes

This project includes several specialized packages for topological data analysis and time series processing:

- **giotto-tda (gtda)**: Topological data analysis toolkit for machine learning
- **ts2vg**: Time series to visibility graph conversion  
- **gudhi**: Geometry Understanding in Higher Dimensions library
- **keras-tuner**: Hyperparameter tuning for Keras models (replaces kerashypetune)

These packages require Python 3.12 for optimal compatibility and have been tested to work together seamlessly.

## Dataset Compatibility

The framework is designed to work with multiple time series datasets, including:
- Human Activity Recognition (HAR)
- ECG/EEG signal classification
- Industrial sensor data
- Environmental monitoring time series


## Examples
The repository includes two example notebooks:

- ts_embed_example1.ipynb: Demonstrates basic time series embedding techniques
- ts_embed_example2.ipynb: Shows more advanced embedding methods and their evaluation


## Environment Testing

After installation, you can verify that your environment is properly set up by running:

```bash
python test_environment.py
```

This script will:
- Check that you have Python 3.12.x
- Verify all required packages are installed and importable
- Test basic functionality of key packages (gtda, ts2vg, gudhi)
- Provide a comprehensive report of your environment status

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.