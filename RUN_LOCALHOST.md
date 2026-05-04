# Run on Localhost (Windows)

This app has two components that must run simultaneously:
- **Backend** — Flask server on `http://localhost:5000`
- **Frontend** — Trame/aiohttp server on `http://localhost:8081`

---

## Prerequisites

Make sure you have Python installed and a virtual environment set up:

```powershell
# From the project root (UGP_FAULT_VISUALISATION_USING_VTK/)
python -m venv .venv
.venv\Scripts\activate
pip install flask segyio numpy vtk trame trame-vuetify trame-vtk aiohttp requests werkzeug
```

> The pretrained model file (`backend/model/pretrained_model.hdf5`) is **not** tracked by git.
> Make sure it is present before running. Ask your supervisor for the file if missing.

---

## Step 1 — Activate the virtual environment (both terminals)

```powershell
cd UGP_FAULT_VISUALISATION_USING_VTK
.venv\Scripts\activate
```

---

## Step 2 — Terminal 1: Start the Backend (Flask, port 5000)

```powershell
cd backend
python backend_server.py
```

You should see:
```
Starting ML & Slice server on http://0.0.0.0:5000
Memory mode: LAZY (only metadata indexed, slices read from disk on demand)
```

---

## Step 3 — Terminal 2: Start the Frontend (Trame, port 8081)

```powershell
cd frontend
python main.py
```

You should see:
```
Starting Seismic Portal Pro on http://localhost:8081
Communicating with Backend Server at: http://localhost:5000
```

---

## Step 4 — Open the App

Open your browser and go to:
```
http://localhost:8081
```

---

## Optional: Point frontend to a different backend

If the backend is running on a remote server (e.g., `172.21.21.158`), set the environment variable before starting the frontend:

```powershell
$env:BACKEND_URL = "http://172.21.21.158:5000"
python main.py
```

---

## Ports Summary

| Component | Host         | Port |
|-----------|--------------|------|
| Backend   | localhost    | 5000 |
| Frontend  | localhost    | 8081 |
