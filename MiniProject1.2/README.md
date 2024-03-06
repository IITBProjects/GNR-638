# GNR638 MiniProject 1: Classification of the Caltech-UCSD Birds-200-2011

## Getting Started

### Dependencies

- Python

### Project Structure

- `codes` : This directory contains all the project implementations.
- `codes/pipeline.py` : This file implements dataset creation and training.
- `codes/dataset.py` : It creates the CUB dataset using the dataframes provided in text files from the `CUB_200_2011` dataset folder.
- `codes/models.py` : Defines all the models using PyTorch.
- `codes/utils.py` : Contains essential helper functions.
- `main.py` : This is the entry point to execute the pipeline functions.
- `setup_env.py` : Sets up the environment for the project.
- `config.json` : Holds parameters for code execution.
- `dataset` : Folder containing the CUB dataset.
- `outputs` : Output folder containing logs, results and plots for each model.

### Installation

To set up the environment, run:
```
python3 setup_env.py
```

### Running the Program

Execute the following command:
```
python3 -u main.py > logs.txt
```

## Authors

- Margav Mukeshbhai Savsani - 200050072  
- Sartaj Islam - 200050128