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

The main dependencies for this project are:
- Python 3.7+
- NumPy
- Pandas
- PyTorch
- Scikit-learn
- UMAP
- Matplotlib
- Seaborn

All dependencies are specified in the `environment.yml` file.


## Installation

### Prerequisites

Ensure you have [Conda](https://docs.conda.io/en/latest/) installed on your system.

### Setting up the environment

```bash
# Clone the repository
git clone https://github.com/imics-lab/time-series-embedding.git
cd time-series-embedding

# Create and activate the Conda environment
conda env create -f environment.yml
# Dynamically set Python path for the environment
conda activate time-series-embedding
conda env config vars set PYTHONPATH=$(pwd):$(pwd)/src
```


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


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.