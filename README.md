# Probabilistic Forecasting of Energy Time Series Using Deep Learning
Repository for master's thesis "Probabilistic Forecasting of Energy Time Series Using Deep Learning" by Yannick Tanner.

## Abstract
TODO

## Usage
The code for the application is located in code and is written in Python.
### Environment
In the code folder there is an anaconda ([anaconda.com](https://www.anaconda.com/)) environment file that contains the dependencies for the project. It can be created via
```
conda env create -f environment.yml
```
The name of the environment is probForecETS and should be used to execute the code of this repository in.

### Data
This software uses data from Open Power Systems Data, because of licensing reasons we refrained from adding this data to the repository. To use it the time_series_15min_singleindex.csv from [https://doi.org/10.25832/time_series/2019-06-05](https://doi.org/10.25832/time_series/2019-06-05) needs to be placed in this folder.

### Load Forecasting
the main files for the load forecasting scenario from the thesis can be found in code/load_forecasting. To execute them the working directory must be the code directory. main.py loads and evaluates the trained models on the data, main_bo.py is the code used for hyper-parameter optimization of the methods.

## Structure
The model code can be found in code/models as well as some wrappers used to incorporate the PyTorch code with scikit-learn and scikit-optimize. The trained models are located under code/trained_models and can be loaded via the skorch wrappers as it is done in code/load_forecasting/main.py.
