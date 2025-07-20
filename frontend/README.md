# Neurocheck Frontend

<b>This is the Streamlit UI for Neurocheck, allowing users to upload EEG or CSV files to get fatigue predictions.</b>

<u>Current status:</u>

- Shows demo / fake predictions if the backend is offline.

- Will connect to a real backend API for predictions when deployed.

---

## Technology Stack

### General Requirements

- `python==3.10.6`

- `streamlit==1.28.0`

### For Data Handling and Visualization

- `numpy`

- `pandas`

- `plotly`

### API communication

- `requests==2.31.0`


<b><i>Currently deployed on Railway platform</i></b>

---

## Quick Start

### Run Locally

#### Navigate to the `frontend` directory

#### Install dependencies

If you have `pip` installed:

```bash
pip install -r requirements.txt
```

#### Run streamlit app

```bash
streamlit run app.py
```
#### Open in browser

---

### Deployment on Railway

The app is currently deployed with the following `Procfile` command:

```bash
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

See `Procfile.notes` for details.


The Python version is pinned and instructs Railway using `runtime.txt`.

---

### Backend Connection

- The frontend calls the backend API at `http://localhost:8000` by default (as instructed in `api_client.py`).

- If the backend is offline, demo predictions are shown instead.

- Update `API_BASE_URL` to the deployed backend URL when ready.

## In Development: Backend API Handshake and Transaction

- Endpoint `/predict/eeg` accepts file uploads via `POST`.

  <i>Currently accepting `.CSV`, `.EDF` with <b>200MB limit</b>.</i>

- Backend returns JSON with prediction `result`:

```json
{
    "backend_status": "online",
    "fatigue_class": "fatigued",
    "confidence": 0.87,
    "filename": filename,
    "note": "You are receiving a test response placeholder."
}
