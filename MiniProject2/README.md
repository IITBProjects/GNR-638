# GNR638 MiniProject 2: Image Deblurring

## Getting Started

### Dependencies

- Python

### Project Structure

- `models` : This directory contains all the model checkpoints.
- `plots` : This directory contains the training plots.
- `config.json` : Holds parameters for code execution.
- `main.py` : The main file for training and evaluation
- `eval.py` : The file to be used for evaluation
- `setup_env.py` : Sets up the environment for the project.

### Installation

To set up the virtual environment, run:
```
python3 setup_env.py
```

### Running the Program for Evaluation

Create the custom_test directory(in the main directory) as was given in the evaluation script (containing blur and sharp folders). Then execute the following command which will create a new folder deblur inside the custom_test directory containing the deblured images gernerated by our model and will also print the PSNR:
```
python3 eval.py
```

## Authors
- Ayush Patil - 200070012
- Margav Mukeshbhai Savsani - 200050072  
- Sartaj Islam - 200050128
