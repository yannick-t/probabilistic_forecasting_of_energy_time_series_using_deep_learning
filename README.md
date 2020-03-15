# Probabilistic Forecasting of Energy Time Series Using Deep Learning
Repository for master's thesis "Probabilistic Forecasting of Energy Time Series Using Deep Learning".
The thesis can be found [here](./thesis.pdf).

Copyright and Licence information for the code adapted from other sources can be found in the respective files as comments.

## Abstract
Deep Learning methods are widely successful and continue to be applied in new fields, one of them being forecasting of energy time series. However, they lack an indication of their confidence in predictions, which helps evaluate, interpret, and improve forecasters.

Therefore, this thesis explores probabilistic extensions to Deep Learning and their application in forecasting of energy time series. The methods tested are Concrete Dropout, Deep Ensembles, Bayesian Neural Networks, Deep Gaussian Processes, and Functional Neural Processes.

For evaluation, two load forecasting scenarios are considered: Short-Term (single-step) and day-ahead (multi-step) forecasting. The methods are evaluated in terms of calibration, sharpness, and how well they indicate a lack of knowledge (Epistemic Uncertainty). As reference a simple Neural Network and a Quantile Regression model are used.

Overall, the methods perform well, with Concrete Dropout, Deep Ensembles, and Bayesian Neural Networks performing similarly well or better to the reference methods. Functional Neural Processes and Deep Gaussian Processes perform worse, likely due to a lack of convergence and sub-optimal parameters. Deep Ensembles in particular prove to be simple to implement and train and to use very few hyper-parameters. Concrete Dropout and Bayesian Neural Networks show similar advantages but need additional configuration for good Epistemic Uncertainty estimates. Furthermore, Concrete Dropout and Deep Ensembles are comparatively fast in training and predictions.

## Usage
The code for the application is located in *code* and is written in Python.
### Environment
In the code folder there is an anaconda ([anaconda.com](https://www.anaconda.com/)) environment file that contains the dependencies for the project. It can be created via (assuming the user is in the repository main directory)
```
conda env create -f code/environment.yml
```
The name of the environment is probForecETS and should be used to execute the code of this repository in.
Activate the environment with
```
conda activate probForecETS
```

### Data
This software uses data from Open Power Systems Data, because of licensing reasons we refrained from adding this data to the repository. To use it, the **time_series_15min_singleindex.csv** from [https://doi.org/10.25832/time_series/2019-06-05](https://doi.org/10.25832/time_series/2019-06-05) needs to be placed in the folder containing the repository data.

Resulting in the following folder structure:
- /
  - probabilistic_forecasting_of_energy_time_series_using_deep_learning
  - time_series_15min_singleindex.csv

### Load Forecasting
the main files for the load forecasting scenario from the thesis can be found in code/load_forecasting. To execute them the working directory must be the *code* directory. *main_opsd_all.py* loads and evaluates the trained models on the data and saves the results in the results folder (will replace files already saved). By default this will recalibrate the models, to modify behavior the *main* method in *main_opsd_all.py* can be modified accordingly. (Note that Functional Neural Processes need a considerable amount of memory, in particular to recalibrate the day-ahead forecasting scenario. They can fail when running out of memory. In tests the process used around 20GB of GPU memory)

Run by switching to the code directory, setting the PYTHONPATH, and calling the interpreter with the main file: (assuming the conda environment was activated)
```
cd code
```
for Linux:
```
PYTHONPATH=./ python load_forecasting/main_opsd_all.py
```
for Windows:
```
set PYTHONPATH=./ && python load_forecasting/main_opsd_all.py
```
Or by using an IDE and configuring it to run *main_opsd_all.py* using code/ as the working directory and probForecETS as the conda environment (if supported otherwise with the python interpreter set to the one from the probForecETS environment).

*main_opsd_optimization.py* is utility code used for hyper-parameter optimization of the methods (not fully automatic).

## Structure
The model code can be found in *code/models* as well as some wrappers used to incorporate the PyTorch code with scikit-learn and scikit-optimize. The trained models are located under *trained_models* and can be loaded via the skorch wrappers as it is done in *code/load_forecasting/main_opsd_all.py*.
