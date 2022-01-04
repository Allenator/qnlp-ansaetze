# Encoding Opposition With Variational Quantum Compositional Models: Supplementary Code

Allen Mi - January 2022

## How-To-Run

The project is developed under `x86-64` Linux. The system dependencies are as follows:

- `conda`: Anaconda or Miniconda, for managing the appropriate Python virtual environment

To install the Python dependencies, run
```
conda env create -f requirements.yml
```
at the root directory of the folder, followed by
```
conda activate qnlp
```
to activate the virtual environment.

## Project structure

- `corpus/`: corpus-related files
  - `corpus.pickle`: pickled 40-sentence corpus
- `data/`: saved training data
  - `all_results.pickle`: pickled training results used in the notebooks
  - `sweep.pickle`: pickled parameter sweep results used in the notebooks
- `figures/`: paper figures
  - omitted
- `ansaetze.py`: implementations for the variational ansaetze
- `utils.py`: utility code for training and testing the model
- `training-and-testing.ipynb`: Jupyter notebook for implementing, training and testing the model
- `make-figures.ipynb`: Jupyter notebook for generating the paper figures
- `requirements.yml`: `conda` virtual environment specifications
- `README.md`: this Markdown file
