# UGP Fault Visualisation Using VTK

This repository contains a small project that identifies faults in 3D seismic volumes using an ML model and provides a web UI (NiceGUI) for uploading SEG-Y files, running inference, and downloading results.

---

**Repository layout**

- `data/` : Workspace input/output area used at runtime.
	- `uploads/` : Place user-uploaded SEG-Y files here (or let the app save uploads).
	- `outputs/` : Inference outputs are saved here (e.g., `faults_<original>.segy`).

- `UGP_FAULT_VISUALISATION_USING_VTK/` : Front-end / app folder containing the NiceGUI app and static assets used for local development.
	- `main.py` : The NiceGUI application entrypoint. Starts a small web server with the UI for uploading SEG-Y files, showing progress, and offering downloads.
	- `ml_core.py` : Contains the ML-related helper functions such as `load_model()` and `process_segy()` used by `main.py` to run inference on a SEG-Y file.
	- `index.html` : A static HTML placeholder. The interactive UI is served by the Python app (`main.py`) — opening `index.html` directly will not start the server.
	- `README.md` : This file.

- `model/` : Saved model artifacts used by `ml_core.py`.
	- `pretrained_model.hdf5` : Keras/TensorFlow weights file (used by `load_model()`).
	- `model3.json` : Model architecture JSON (optional depending on how `load_model()` is implemented).

---

**What each key file does**

- `main.py` — application flow:
	- Initializes folders (`data/uploads`, `data/outputs`).
	- Loads the ML model once (cached) via `ml_core.load_model()`.
	- Exposes a NiceGUI interface to upload a `.segy`/`.sgy` file, shows processing progress, and provides download links to results.
	- Uses `run.cpu_bound(process_segy, input_path, output_path, model)` to run the heavy ML work off the event loop.

- `ml_core.py` — ML helpers (expected responsibilities):
	- `load_model()` : Load the model from files in `model/` and return a callable model object.
	- `process_segy(input_path, output_path, model)` : Read the input SEG-Y, run model inference (patch-wise or whole-volume), write a fault-segmentation SEG-Y to `output_path`.

- `index.html` — static placeholder used when viewing files directly. The app is intended to be run via `main.py` which hosts the real UI in the browser.

---

**How to run (local)**

1. Create a virtual environment (recommended) and install dependencies. Example using Python 3.10+:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install nicegui numpy scipy tensorflow vtk
```

2. Start the app from the `UGP_FAULT_VISUALISATION_USING_VTK` folder:

```powershell
cd "UGP_FAULT_VISUALISATION_USING_VTK"
python main.py
```

3. Open the UI in your browser at http://localhost:8080 (NiceGUI default).

4. Upload a SEG-Y file via the web UI. After processing, download the output from the `outputs/` folder or use the provided download button.

---

**Notes, assumptions and troubleshooting**

- The repository does not ship a bundled JavaScript `main.js` file — the UI is server-rendered by `main.py`. Do not open `index.html` directly expecting the app to be functional.
- Ensure the `model/` files exist and that `ml_core.load_model()` knows how to load them. If you see errors about missing model files, check `model/pretrained_model.hdf5` and `model3.json`.
- Large SEG-Y files can take significant memory and CPU; the app uses a background CPU-bound runner but keep an eye on machine resources.
- If you need to add or update Python dependencies, create a `requirements.txt` with pinned versions.

---

If you want, I can:

- generate a `requirements.txt` based on the imports used by `main.py` and `ml_core.py`;
- add a brief example of `ml_core.py`'s expected `load_model()` and `process_segy()` signatures;
- run the NiceGUI app here to verify startup (I will need permission to run `python main.py`).

---



