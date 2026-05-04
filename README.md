# Fc-Fuse

Fc-Fuse is a TensorFlow-based multi-feature fusion model project for network traffic classification experiments.

## Project Structure

```text
Fc-Fuse/
├── file/                 # Dataset directory, large CSV files are not tracked by Git
├── main/                 # Model, training, evaluation, and data processing code
└── requirements/         # Python dependency list
```

## Requirements

Install dependencies with:

```bash
pip install -r requirements/requirements.txt
```

## Dataset

The original CSV datasets are large and are excluded from this repository:

- `file/CIC_bl_ordered.csv`
- `file/botiot_last_datasets.csv`

Place the required dataset files in the `file/` directory before running training.

## Usage

Run the training entry point from the `main` directory:

```bash
cd main
python main.py
```

By default, `main.py` loads:

```text
../file/botiot_last_datasets.csv
```

Training logs and saved models are written to the `logs/` directory.
