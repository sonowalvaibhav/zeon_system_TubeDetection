# Zeon System

This repository contains scripts for YOLOv8 oriented bounding box training, evaluation, data splitting, and prediction for object localization and orientation detection of 70 overhead images of centrifuge tubes 

## Setup

```powershell
cd "c:\Users\acer\Desktop\zeon system"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Prepare dataset

```powershell
python src/split_data.py
```

### Train model

```powershell
python src/train.py
```

### Quick train test

```powershell
python src/quick_train.py
```

### Evaluate model

```powershell
python src/evaluate.py
```

### Run prediction

```powershell
python src/predict.py
```
