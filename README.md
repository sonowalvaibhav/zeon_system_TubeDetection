# Zeon System

This repository contains scripts for YOLOv8 oriented bounding box training, evaluation, data splitting, and prediction.

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

## GitHub push workflow

1. Install Git for Windows from https://git-scm.com/download/win
2. Restart PowerShell
3. Run:

```powershell
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```
