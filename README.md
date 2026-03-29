# UGP Fault Visualisation Using VTK

This project provides a local web app for:

- uploading SEG-Y seismic volumes
- running ML-based fault extraction
- downloading the generated fault SEG-Y
- viewing seismic data in the 3D workspace

The current UI is built with `trame`

## Project Layout

- `main.py`
  Starts the Trame app and exposes the HTTP endpoints used by the ML tool:
  - `/upload_ml`
  - `/run_ml`
  - `/ml_status`

- `ml_tool.html`
  Standalone browser page embedded into the ML tab with an iframe.
  Handles:
  - file upload
  - upload progress
  - ML inference progress polling
  - output download

- `processing.py`
  Loads the trained model once and runs ML extraction through `ml_core.py`.

- `ml_core.py`
  Core SEG-Y ML pipeline:
  - loads the TensorFlow model
  - reads the input SEG-Y
  - runs patch-wise inference
  - writes the output SEG-Y
  - emits inference progress updates

- `model/`
  Contains model files used by `ml_core.py`:
  - `model3.json`
  - `pretrained_model.hdf5`
  - `pretrained_model.hdf5` is tracked with Git LFS.
  - After cloning, run the following to download the actual weights file:

```bash
git lfs install
git clone <your-repo-url>
cd UGP_FAULT_VISUALISATION_USING_VTK
git lfs pull
```

  - If `model/pretrained_model.hdf5` is missing locally, the ML pipeline will not run.


- `../data/uploads/`
  Stores uploaded SEG-Y input files.

- `../data/outputs/`
  Stores generated output files such as `faults_<input>.segy`.

## Install

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

Start the app from inside the app folder:

```powershell
cd UGP_FAULT_VISUALISATION_USING_VTK
python main.py
```

Open:

```text
http://localhost:8081
```

## ML Workflow

In the `ML Processing` tab:

1. Select a `.segy` or `.sgy` file.
2. Wait for the upload to complete.
3. Click `Run Fault Extraction`.
4. Watch upload progress and ML inference progress.
5. Download the generated output SEG-Y file.

## Notes

- The ML pipeline expects a standard 3D SEG-Y with valid inline/crossline geometry.
- Large files can take time and memory because inference runs patch-by-patch.
- The output SEG-Y is written to `data/outputs/`.
- The app serves the standalone ML tool page from the app folder and serves generated outputs through the web server for download.

## Current Main Dependencies

- `trame`
- `trame-vuetify`
- `aiohttp`
- `tensorflow-cpu`
- `numpy`
- `segyio`
- `vtk`
