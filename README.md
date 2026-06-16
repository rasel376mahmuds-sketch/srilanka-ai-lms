# Sri Lankan LMS AI Model

This project contains the Machine Learning / AI model for a Sri Lankan Learning Management System (LMS).

## Structure
- `data/`: Datasets.
- `models/`: Saved model weights or artifacts.
- `notebooks/`: Jupyter notebooks for data exploration.
- `src/`: Source code including the API.
  - `api/`: FastAPI application to serve the model.
  - `models/`: Model training scripts.
  - `data/`: Data loading/processing scripts.

## Setup
1. Create a virtual environment: `python -m venv venv`
2. Activate the virtual environment: `.\venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Run the API: `cd src` then `uvicorn api.main:app --reload`
