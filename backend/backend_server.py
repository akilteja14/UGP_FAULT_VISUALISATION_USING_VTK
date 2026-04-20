"""
backend_server.py
=================
Unified Flask backend for the Seismic Portal Pro system.

This server has two primary responsibilities:
  1. SLICE SERVER (lazy, memory-efficient):
     - On /load  : Reads ONLY the trace header metadata from the SEG-Y file and
                   builds a lightweight 'trace_map' lookup table. The full 3D
                   amplitude data is NOT loaded into RAM.
     - On /slice : Opens the SEG-Y file on-demand, uses trace_map to locate
                   exactly the right traces, reads just those traces from disk,
                   and returns them as a 2D array. After serving, the file is
                   closed immediately to free the file handle.

  2. ML SERVER:
     - /upload_ml : Accepts an uploaded SEG-Y file and saves it to disk.
     - /run_ml    : Triggers the ML fault-extraction pipeline in a background thread.
     - /ml_status : Reports real-time progress of the running ML job.
     - /download  : Allows the frontend to download the generated output SEG-Y.

MEMORY COMPARISON (before vs after this implementation):
  Before: A 1000x1000x1000 float32 volume = ~4 GB loaded into RAM at all times.
  After:  Only trace_map (1000x1000 int32) = ~4 MB in RAM. Full cube never loaded.
"""

import os
import threading
import uuid

import numpy as np
import segyio
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

import processing

app = Flask(__name__)

# =============================================================================
# -- Global Volume State (Lazy Index, NOT the Full Cube) --
# =============================================================================
# This lock ensures that concurrent requests (e.g., multiple slice requests
# arriving at the same time) don't interfere with each other when reading
# or updating the shared volume metadata below.
volume_lock = threading.Lock()

# Path to the currently loaded SEG-Y file on disk.
# Slices are read directly from this file on each request.
loaded_path = ""

# Shape of the full 3D volume: (n_inlines, n_crosslines, n_samples)
# This tells the frontend the dimensions of the cube so it can set slider ranges.
volume_shape = None

# trace_map[i, j] = the global trace index in the SEG-Y file that corresponds
# to inline index i and crossline index j.
# This is the core of the lazy approach: we only store a 2D int matrix (~4 MB)
# instead of the full 4 GB floating-point cube.
trace_map = None

# Arrays of the actual inline and crossline numbers (not just their count).
# Used to convert a requested index (e.g., iline=50) to the SEG-Y header value.
unique_inlines = None
unique_crosslines = None

# =============================================================================
# -- ML Server State --
# =============================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dictionary tracking the state of the currently running (or last run) ML job.
ml_job_status = {
    "stage":             "idle",
    "message":           "Awaiting File...",
    "percent":           0,
    "running":           False,
    "completed_patches": 0,
    "total_patches":     0,
    "elapsed_seconds":   0,
    "eta_seconds":       0,
    "output_path":       "",
}


# =============================================================================
# -- Lazy Metadata Indexing --
# =============================================================================

def index_volume(file_path):
    """
    LAZY INDEXING: Read only the trace header attributes from the SEG-Y file
    to build a lookup table (trace_map) without loading any amplitude data.

    For a 1000x1000x1000 volume this replaces loading ~4 GB of float32 data
    with reading only ~12 MB of integer header data.

    The resulting trace_map[i, j] = k means:
      - The trace at inline index i and crossline index j
        is stored at position k in the SEG-Y file.
      - When a slice is later requested, we simply open the file, seek to k,
        and read one trace of amplitudes.

    Args:
        file_path (str): Absolute path to the SEG-Y file.

    Updates global state: loaded_path, volume_shape, trace_map,
                          unique_inlines, unique_crosslines.

    Returns:
        dict: Metadata with the file path and volume shape, sent back to frontend.
    """
    #telling to use the global variables defined above
    global loaded_path, volume_shape, trace_map, unique_inlines, unique_crosslines

    #gives absolute path to the file_path(output segy file)...
    normalized_path = os.path.abspath(file_path)

    #effectively if path doesn't exist the file doesn't exist...
    if not os.path.exists(normalized_path):
        raise FileNotFoundError(f"SEG-Y file not found: {normalized_path}")

    #we always do ignore_geometry = true because we want to build the cube
    #geometry on our own, if we use segyio geometry construction it raises
    #errors for every small mistake(like header doesn't exist for a trace or so)
    with segyio.open(normalized_path, ignore_geometry=True) as segy_file:
        # -- Read inline and crossline indices from trace headers --
        # We try the standard TraceField enum first, then fall back to raw
        # byte offsets (181 and 185) for non-standard SEG-Y files.
        try:
            #creating inlines and crosslines array.
            inlines    = segy_file.attributes(segyio.TraceField.INLINE_3D)[:]
            crosslines = segy_file.attributes(segyio.TraceField.CROSSLINE_3D)[:]
        except Exception:
            #for non standard segy files
            inlines    = segy_file.attributes(181)[:]
            crosslines = segy_file.attributes(185)[:]

        # -- Discover the set of unique inline and crossline values --
        uq_inlines    = np.unique(inlines)
        uq_crosslines = np.unique(crosslines)
        n_samples     = len(segy_file.samples)

        # -- Build reverse lookup dictionaries for O(1) index resolution --
        # a map where key is inline_val and key-value is inline_idx to get the range
        # of inline values to be continuous values.
        inline_index    = {val: idx for idx, val in enumerate(uq_inlines)}
        crossline_index = {val: idx for idx, val in enumerate(uq_crosslines)}

        # -- Build the trace_map: a 2D integer grid --
        # trace_map[i, j] holds the global trace number in the file.
        # -1 means that particular (inline, crossline) combination has no trace.
        new_trace_map = np.full(
            (len(uq_inlines), len(uq_crosslines)),
            fill_value=-1,
            dtype=np.int32
        )

        # Walk through every trace in the file (header-only, no amplitude data)
        # and populate the trace_map with the trace's position in the file.

        #continue....
        #way to storing new_trace_map[i, j] = trace_id 
        for trace_id in range(segy_file.tracecount):
            il  = inlines[trace_id]
            xl  = crosslines[trace_id]
            i   = inline_index.get(il)
            j   = crossline_index.get(xl)
            if i is not None and j is not None:
                new_trace_map[i, j] = trace_id

    # -- Update global state under the lock --
    # so that concurrent access cannot corrupt the global variables
    with volume_lock:
        loaded_path       = normalized_path
        volume_shape      = (len(uq_inlines), len(uq_crosslines), n_samples)
        trace_map         = new_trace_map
        unique_inlines    = uq_inlines
        unique_crosslines = uq_crosslines

    return {"path": loaded_path, "shape": list(volume_shape)}


# =============================================================================
# -- On-Demand Slice Reading --
# =============================================================================

# Reads one inline slice from the loaded SEG-Y file.
# For a fixed inline index, it reads all crossline traces across all time samples
# using trace_map and returns a 2D array of shape crosslines x samples.
def read_inline_slice(slice_index):
    """
    Read a single INLINE slice directly from the SEG-Y file on disk.

    An inline slice is a vertical plane cut along the inline axis, returning
    a 2D array of shape (n_crosslines, n_samples).

    This is memory-efficient because we only read n_crossline traces instead
    of loading the entire volume. The file is opened and closed within
    this function call so we don't hold a persistent file descriptor.

    Args:
        slice_index (int): The position in the inline dimension (0-based index).

    Returns:
        np.ndarray: 2D float32 array of shape (n_crosslines, n_samples).

    Raises:
        ValueError: If no volume is loaded yet.
    """
    n_crosslines = volume_shape[1]
    n_samples    = volume_shape[2]
    result       = np.zeros((n_crosslines, n_samples), dtype=np.float32)

    with segyio.open(loaded_path, ignore_geometry=True) as segy_file:
        for j in range(n_crosslines):
            trace_id = trace_map[slice_index, j]
            if trace_id >= 0:
                # Read just this one trace from disk (seek + read, not full scan)
                result[j, :] = segy_file.trace[trace_id]

    return result


# Reads one crossline slice from the loaded SEG-Y file.
# For a fixed crossline index, it reads all inline traces across all time samples
# and returns a 2D array of shape inlines x samples.
def read_crossline_slice(slice_index):
    """
    Read a single CROSSLINE slice directly from the SEG-Y file on disk.

    A crossline slice is a vertical plane cut along the crossline axis, returning
    a 2D array of shape (n_inlines, n_samples).

    Args:
        slice_index (int): The position in the crossline dimension (0-based index).

    Returns:
        np.ndarray: 2D float32 array of shape (n_inlines, n_samples).
    """
    n_inlines = volume_shape[0]
    n_samples = volume_shape[2]
    result    = np.zeros((n_inlines, n_samples), dtype=np.float32)

    with segyio.open(loaded_path, ignore_geometry=True) as segy_file:
        for i in range(n_inlines):
            trace_id = trace_map[i, slice_index]
            if trace_id >= 0:
                result[i, :] = segy_file.trace[trace_id]

    return result

# Reads one horizontal time/depth slice from the loaded SEG-Y file.
# For a fixed sample index, it reads that sample from every inline-crossline trace
# and returns a 2D array of shape inlines x crosslines.
def read_time_slice(slice_index):
    """
    Read a single TIME slice directly from the SEG-Y file on disk.

    A time slice is a horizontal plane at a fixed depth/time sample, returning
    a 2D array of shape (n_inlines, n_crosslines). Unlike inline/crossline slices,
    this requires reading ONE sample from EVERY trace in the file, making it the
    most disk-intensive of the three slice modes. However, it is still far more
    efficient than loading the full cube into RAM.

    Args:
        slice_index (int): The position in the time/depth dimension (0-based index).

    Returns:
        np.ndarray: 2D float32 array of shape (n_inlines, n_crosslines).
    """
    n_inlines    = volume_shape[0]
    n_crosslines = volume_shape[1]
    result       = np.zeros((n_inlines, n_crosslines), dtype=np.float32)

    with segyio.open(loaded_path, ignore_geometry=True) as segy_file:
        for i in range(n_inlines):
            for j in range(n_crosslines):
                trace_id = trace_map[i, j]
                if trace_id >= 0:
                    # Read only the single sample at the requested depth index
                    result[i, j] = segy_file.trace[trace_id][slice_index]

    return result


# =============================================================================
# -- Flask Endpoints: Volume Management --
# =============================================================================

@app.route("/status", methods=["GET"])
def status():
    """
    Health-check endpoint. Reports whether a volume is currently indexed
    and ready to be sliced, and what its dimensions are.

    Returns:
        JSON: { loaded: bool, path: str, shape: [int, int, int] or null }
    """
    with volume_lock:
        return jsonify({
            "loaded": loaded_path != "",
            "path":   loaded_path,
            "shape":  list(volume_shape) if volume_shape else None,
        })

#if load endpoint is triggered via a HTTP POST request run this function 
@app.route("/load", methods=["POST"])
def load():
    """
    Indexes a SEG-Y volume file WITHOUT loading it into RAM.

    Expects JSON body: { "path": "/absolute/path/to/file.segy" }

    Runs index_volume() which only reads trace headers, builds the trace_map
    lookup table, and stores the volume shape. Returns instantly (seconds,
    not minutes) because no amplitude data is read.

    Returns:
        JSON: { path: str, shape: [n_inlines, n_crosslines, n_samples] }
    """
    #receives the output segy file as json from the function
    #configure_slice_server with path as key

    #silent = true -> invalid json no error but payload becomes none or {}
    payload   = request.get_json(silent=True) or {}
    #strip removes trailing and leading whitespace (just for safety)
    file_path = (payload.get("path") or "").strip()

    #if file path is none we send an error json message and HTTP status code
    #400 indicating it is an error to configure_slice_server function
    if not file_path:
        return jsonify({"error": "Missing SEG-Y path."}), 400

    #continue....
    try:
        #gets metadata from the function index_volume
        metadata = index_volume(file_path)
    except Exception as err:
        #if any error sends an HTTP post request 500 indicating an error
        #sends json of the error message
        return jsonify({"error": str(err)}), 500

    #returns json of metadata
    return jsonify(metadata)

#run this function when it recieves an get request to /slice endpoint from frontend 
@app.route("/slice", methods=["GET"])
def get_slice():
    """
    Serves a single 2D plane from the seismic volume by reading it directly
    from the SEG-Y file on disk at the moment of request.

    NO full volume cube is kept in RAM. Only the requested plane (a few MB)
    is assembled and returned, then immediately garbage-collected.

    Query parameters:
      - mode      : "inline" | "crossline" | "time"
      - inline    : (int) Position along the inline axis (used for "inline" mode)
      - crossline : (int) Position along the crossline axis (used for "crossline" mode)
      - time      : (int) Position along the time/depth axis (used for "time" mode)

    Returns:
        JSON: { shape: [int, int], data: [[...], ...] }
    """
    with volume_lock:
        #if there is no loaded_path or trace_map is none then return error
        #with HTTP STATUS 400 to frontend
        if loaded_path == "" or trace_map is None:
            return jsonify({"error": "No SEG-Y volume loaded. Call /load first."}), 400

        #using default "mode" as inline and we use "mode" from the
        #json we recieve from frontend...
        mode = request.args.get("mode", "inline")

    try:
        if mode == "inline":
            #Vertical plane: cut along all crosslines and time for a fixed inline
            #get's inline value from "inline" key in json...
            inline_idx = int(request.args.get("inline", 0))
            #continue...
            #2d numpy array of seiemic ampltiudes along all crosslines and time samples
            data = read_inline_slice(inline_idx)

        elif mode == "crossline":
            #Vertical plane: cut along all inlines and time for a fixed crossline
            #get's crossline value from "crossline" key in json...
            crossline_idx = int(request.args.get("crossline", 0))
            #similarily...
            data = read_crossline_slice(crossline_idx)

        elif mode == "time":
            #Horizontal plane: a snapshot of the entire survey at one time sample
            #get's time value from "time" key in json...
            time_idx = int(request.args.get("time", 0))
            #similarily...
            data = read_time_slice(time_idx)

        else:
            #unsupported slice mode error returned to frontend with HTTP STATUS code 400,
            #to fetch_slice function...
            return jsonify({"error": f"Unsupported slice mode: '{mode}'"}), 400

    except Exception as err:
        #if any error returns HTTP STATUS code 500 to the frontend...
        return jsonify({"error": str(err)}), 500

    #if everything works as expected we send shape and data(as list) to frontend
    #fetch_slice function in json format
    return jsonify({"shape": list(data.shape), "data": data.tolist()})


# =============================================================================
# -- Flask Endpoints: ML Pipeline --
# =============================================================================

@app.route("/upload_ml", methods=["POST"])
def upload_ml():
    """
    Accepts a multipart file upload of a SEG-Y file from the frontend.
    Saves the file to UPLOAD_DIR with a unique suffix to prevent filename collisions.

    Returns:
        JSON: { saved_path: str, filename: str }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    original_name = secure_filename(file.filename)
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in [".segy", ".sgy"]:
        return jsonify({"error": "Only .segy and .sgy files are supported."}), 400

    # Append a short UUID to prevent overwriting files with the same name
    stem       = os.path.splitext(original_name)[0]
    saved_name = f"{stem}_{uuid.uuid4().hex[:8]}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)

    file.save(saved_path)
    return jsonify({"saved_path": saved_path, "filename": saved_name})


@app.route("/run_ml", methods=["POST"])
def run_ml():
    """
    Triggers the fault extraction ML pipeline in a background thread.

    The ML job runs asynchronously so this endpoint returns immediately
    with a "queued" response. The frontend polls /ml_status to track progress.

    Expects JSON body: { "input_path": "/path/to/uploaded/file.segy" }

    Returns:
        JSON: { output_path: str, message: str }
    """
    payload    = request.get_json(silent=True) or {}
    input_path = os.path.abspath((payload.get("input_path") or "").strip())

    if not input_path or not os.path.exists(input_path):
        return jsonify({"error": "Missing or invalid input file path."}), 400

    output_path = processing.build_ml_output_path(input_path, OUTPUT_DIR)

    def progress_callback(update):
        """
        Called by the ML core on every progress update tick.
        Updates the shared ml_job_status dict so /ml_status can report it.
        """
        ml_job_status.update(update)
        # Mark as no longer running when the stage transitions to "complete"
        ml_job_status["running"] = update.get("stage") != "complete"

    def ml_worker():
        """
        Background thread target. Runs the full ML extraction pipeline and
        updates ml_job_status to 'complete' or 'error' when done.
        """
        try:
            processing.run_ml_extraction(input_path, output_path, progress_callback)
            ml_job_status.update({
                "stage":   "complete",
                "message": "Process Finished. Success!",
                "percent": 100,
                "running": False,
                "output_path": output_path,
            })
        except Exception as err:
            ml_job_status.update({
                "stage":   "error",
                "message": str(err),
                "running": False,
            })

    # Reset the status dict before launching the thread
    ml_job_status.update({
        "stage":             "queued",
        "message":           "Queued for processing...",
        "percent":           0,
        "running":           True,
        "completed_patches": 0,
        "total_patches":     0,
        "elapsed_seconds":   0,
        "eta_seconds":       0,
        "output_path":       "",
    })

    threading.Thread(target=ml_worker, daemon=True).start()

    return jsonify({
        "output_path": output_path,
        "message":     "Started ML process in background."
    })


@app.route("/ml_status", methods=["GET"])
def get_ml_status():
    """
    Returns the current status of the ML job as a JSON object.
    The frontend polls this endpoint every ~1.5 seconds to update its progress bar.

    Returns:
        JSON: { stage, message, percent, running, completed_patches,
                total_patches, elapsed_seconds, eta_seconds, output_path }
    """
    return jsonify(ml_job_status)


@app.route("/download/<path:filename>", methods=["GET"])
def download_file(filename):
    """
    Serves a generated output SEG-Y file as a binary download attachment.
    The frontend provides a link pointing here so the user can save the
    fault-extracted result locally.

    Args:
        filename (str): Name of the file inside OUTPUT_DIR to download.

    Returns:
        File response as an attachment.
    """
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


# =============================================================================
# -- Entry Point --
# =============================================================================

if __name__ == "__main__":
    print("Starting ML & Slice server on http://0.0.0.0:5000")
    print("Memory mode: LAZY (only metadata indexed, slices read from disk on demand)")
    app.run(host="0.0.0.0", port=5000)
