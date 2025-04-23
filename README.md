# Reinforcement Learning S25 Project

Dmitry Beresnev / <d.beresnev@innopolis.university>, 

Vsevolod Klyushev / <v.klyushev@innopolis.university> 

Nikita Yaneev / <n.yaneev@innopolis.university>

## Projects description

We propose a new approach that leverages reinforcement learning (RL) to generate regular expression.
This method formulates the regular expression generation process as a Markov decision process (MDP)
with fully-observable deterministic episodic environment.

## Requirements

Code was tested on Windows 11, Python 3.12

All the requirement packages are listed in the file `pyproject.toml`

## Before start

Using [uv](https://docs.astral.sh/uv/):
```
uv sync
```

Optionally setup pre-commit hook:
```
uv run pre-commit install
```

and test it:
```
uv run pre-commit run --all-files
```

We also highly recommend reading report to fully understand context and purpose of some files and folders.

## Repository structure

```text
├── data                            # Data used for experiments
├───── email.json                   # Data for experiment "Simple Email"
├───── single_number.json           # Data for experiment "Single Number"
├───── word.json                    # Data for experiment "[cat] Words"
|
├── report                  
├───── pictures
├──────── *.jpg, *.png
├───── main.pdf
├───── main.tex
|
├── src                             # Source notebooks and scripts
├───── notebooks
├──────── a2c_email.ipynb           # A2C for "Simple Email" experiment
├──────── a2c_number.ipynb          # A2C for "Single Number" experiment
├──────── a2c_word.ipynb            # A2C for "[cat] Words" experiment
├──────── generate_dataset.ipynb    # Data for dataset generation
├──────── reinforce_email.ipynb     # Reinforce for "Simple Email" experiment
├──────── reinforc_number.ipynb     # Reinforce for "Single Number" experiment
├──────── reinforc_word.ipynb       # Reinforce for "[cat] Words" experiment
├───── dataset.py
├───── environment_metrics.py       # Environment with Metrics Approach reward
├───── environment.py               # Environment with XOR Approach reward
├───── rpn.py                       # Reverse Polish Notation implementation
|
├── .pre-commit-config.yaml
├── .python-version
├── pyproject.toml       # Formatter and linter settings
├── README.md            # The top-level README
|
└── uv.lock              # Information about uv environment
```

## Contacts

In case of any questions you can contact us via university emails listed at the beginning
