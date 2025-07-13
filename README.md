# 2025 Summer Internship
ML Pipeline that allows to test different imbalance classification approaches systematically.

## Overview
| Stage | Script | Purpose |
|-------|--------|---------|
| Load CSV | src/data_loader.py | Streams the semicolon-delimited CSV in 50 k-row chunks; keeps every feat_ column; casts features to float32 and the label (Class) to int8. |
| Group split | src/split_data.py | GroupShuffleSplit (80 / 20, random_state) so each Info_group appears only in train or test. |
| Pipeline | src/pipeline.py | sampler → StandardScaler → PCA → classifier; PCA keeps 50 components (~85 % variance). Evaluates accuracy, precision, recall, F1, ROC-AUC. Saves plots. |
| Entry point | main.py | CLI / .env front-end. Lets you choose sampler, classifier, random seed and dataset path. |


##  Set up the environment

Before utilizing this pipeline, ensure that you have activated a Python environment with the specified requirements:

```bash
python -m venv venv
source venv/bin/activate        # Windows - .\venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Pipeline

To execute this pipeline, a single entry point has been provided `main.py` which can be executed in two different ways:
```bash
python main.py
```

or:
```bash
python main.py -default_values
```

If you wish to use the `-default_values` flag, you must first populate the `.env` with the required parameters:

```bash
NAME=<pipeline name>        # Used to identify the run.
FILE_PATH=<csv_file_path>
CLASSIFIER=<classifier>
IMBALANCE_HANDLER=<sampler>
RANDOM_STATE=<random_state>
```

## Results
The pipeline will return the following information:
1. Accuracy
2. Precision
3. Recall
4. F1-Score
5. ROC-AUC
6. Graphs:
    - confusion_matrix_<pipeline_name>.png
    - precision_recall_curve_<pipeline_name>.png
    - roc_curve_<pipeline_name>.png

