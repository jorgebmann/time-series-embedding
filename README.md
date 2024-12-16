


<div align="center">


<!-- TITLE -->
# `A Comprehensive Evaluation of Time Series Embedding Methods for Classification Tasks`

This Repository features the source code for the paper with the same title. For access to the papers, please follow the following links:

// [![arXiv](https://img.shields.io/badge/cs.LG-arXiv:1234.56789-b31b1b.svg)](https://arxiv.org/abs/1234.56789)


</div>


<!-- DESCRIPTION -->
## Description
This repository contains code for the quantitative evaluation of different time series embedding methods based on their theoretical foundations and application contexts.
Unlike previous work, this study quantitatively compares these methods by testing their performance on various classification tasks across different datasets. 
The results reveal that embedding methods perform differently depending on the dataset and classification algorithm, emphasizing the need for careful model selection and experimentation. 
If you plan on using this repository, please cite the relevant paper as:



## Setup


To create the Conda Environment, please make sure to install the following libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `optuna`
- `scikit-learn`
- `pywt`
- `scipy`
- `fastdtw`
- `tensorflow`
- `kerashypetune`
- `gtda`
- `umap-learn`
- `networkx`
- `ts2vg`
- `gudhi`
- 'POT'
- 'PyWavelet'
- 'tabulate'
- 'xgboost'
- 'fastdtw'
- 'giotto-tda'


### Conda Virtual Environment

Create the Conda virtual environment using the [environment file](environment.yml):
```bash
conda env create -f environment.yml

# dynamically set python path for the environment
conda activate YOUR_PROJECT
conda env config vars set PYTHONPATH=$(pwd):$(pwd)/src
```


<!-- USAGE -->
## Usage
To use this code, the preprocessing of the data first needs to take place. Then, you can simply apply different combinations of the embedding methods in this repository and classification methods.
An Example of the code has been also been provided. 

<!-- CONTRIBUTING -->
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.


<!-- CITATION -->
## Citation

To cite this work, please use the following:


```bibtex
@article{YourName,
  title={Your Title},
  author={Your team},
  year={Year}
}
```
