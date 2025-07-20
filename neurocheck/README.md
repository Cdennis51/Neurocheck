# Neurocheck BackEnd

<b>This is the **FastAPI backend** for the NeuroCheck MVP.</b>

<u>Current status:</u>

- A health check endpoint (`/`)

- A placeholder EEG fatigue prediction endpoint (`/predict/eeg`)


<u>Future versions will integrate:</u>

- Preprocessing of EEG/MRI/voice data

- Model inference (classification + visualization outputs)

---

## Technology Stack

### General Requirements

- `python==3.10.6`

- `uvicorn`  (ASGI server for FastAPI)

### API communication

- `fastapi` (core backend framework)

- `pydantic` (data validation)

### ML & Data Handling

- `numpy`

- `pandas`

- `scikit-learn`

- `tensorflow`

- `matplotlib`

- `seaborn`

### Development Tools (Local Only)

- `pytest` (testing framework)

- `jupyterlab` (notebook experiments)

- `pylint` (linting)

---

## Quick Start

### Run Locally

#### Navigate to the Backend (`neurocheck`) directory

#### Install dependencies

If you have `pip` installed:

```bash
pip install -r requirements.txt
```

#### Run the BackEnd Server

Start the backend with Uvicorn:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- `--reload` enables auto-reloading during development.

- The backend is currently accessible at `http://localhost:8000`


- You should see a JSON response:
```json
{
    "message": "Backend is running! Use /predict/eeg for predictions."
}
```

#### API Endpoints

 Method | Endpoint       | Description                                                                                                     |
| ------ | -------------- | --------------------------------------------------------------------------------------------------------------- |
| `GET`  | `/`            | Health check endpoint. Confirms backend is running.                                                             |
| `POST` | `/predict/eeg` | Accepts an uploaded EEG file (CSV/EDF). <br> Returns a **placeholder response** with a dummy fatigue classification. |

#### Example request using curl:

```bash
curl -X POST "http://localhost:8000/predict/eeg" \
     -F "file=@neurocheck/db-in-progress/
     backend/data/dummy_sample_eeg.csv"
```

<u>Expected placeholder response:</u>
```json
{
  "backend_status": "online",
  "fatigue_class": "fatigued",
  "confidence": 0.87,
  "filename": "sample_eeg.csv",
  "note": "You are receiving a test response placeholder."
}
```

---

## Current *Project Structure

```bash
neurocheck/
│
├── api/
│   └── main.py # FastAPI routes (main backend entrypoint)
│
├── ml_logic/   # Future preprocessing & ML model logic
│
├── db-in-progress/ # Contributor dB "Notebook"
│
├── requirements.txt    # Backend dependencies
└── README.md   # This file
```

---

## Temporary Railway Deployment While Awaiting Model Integration (Will Revise and Update)

For Railway, the backend will run with the same Uvicorn command.

```web: uvicorn api.main:app --host 0.0.0.0 --port $PORT --reload```

---

## In Development: API Handshake and Transaction

- Add Procfile for temporary Backend Railway Deployment

-  Confirm backend runs locally

-  Validate /predict/eeg returns placeholder output

-  Add EEG preprocessing module (ml_logic/preprocess.py)

-  Integrate model inference (ml_logic/model.py)

-  Connect backend with Streamlit frontend via utils/api_client.py

-  Deploy backend to Railway
