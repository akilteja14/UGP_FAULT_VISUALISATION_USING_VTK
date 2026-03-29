import os
import threading

import numpy as np
import segyio
from flask import Flask, jsonify, request


app = Flask(__name__)
cube_lock = threading.Lock()
cube = None
cube_shape = None
loaded_path = ""


def build_cube(file_path):
    with segyio.open(file_path, ignore_geometry=True) as segy_file:
        try:
            inlines = segy_file.attributes(segyio.TraceField.INLINE_3D)[:]
            crosslines = segy_file.attributes(segyio.TraceField.CROSSLINE_3D)[:]
        except Exception:
            inlines = segy_file.attributes(181)[:]
            crosslines = segy_file.attributes(185)[:]

        samples = segy_file.samples
        unique_inlines = np.unique(inlines)
        unique_crosslines = np.unique(crosslines)

        inline_index = {value: index for index, value in enumerate(unique_inlines)}
        crossline_index = {value: index for index, value in enumerate(unique_crosslines)}

        data_cube = np.zeros(
            (len(unique_inlines), len(unique_crosslines), len(samples)),
            dtype=np.float32,
        )

        for trace_id in range(segy_file.tracecount):
            inline_value = inlines[trace_id]
            crossline_value = crosslines[trace_id]
            i = inline_index[inline_value]
            j = crossline_index[crossline_value]
            data_cube[i, j, :] = segy_file.trace[trace_id]

    return data_cube


def load_volume(file_path):
    global cube, cube_shape, loaded_path

    normalized_path = os.path.abspath(file_path)
    if not os.path.exists(normalized_path):
        raise FileNotFoundError(normalized_path)

    data_cube = build_cube(normalized_path)

    with cube_lock:
        cube = data_cube
        cube_shape = data_cube.shape
        loaded_path = normalized_path

    return {"path": loaded_path, "shape": list(cube_shape)}


@app.route("/status", methods=["GET"])
def status():
    with cube_lock:
        return jsonify(
            {
                "loaded": cube is not None,
                "path": loaded_path,
                "shape": list(cube_shape) if cube_shape else None,
            }
        )


@app.route("/load", methods=["POST"])
def load():
    payload = request.get_json(silent=True) or {}
    file_path = (payload.get("path") or "").strip()
    if not file_path:
        return jsonify({"error": "Missing SEG-Y path."}), 400

    try:
        metadata = load_volume(file_path)
    except Exception as err:
        return jsonify({"error": str(err)}), 500

    return jsonify(metadata)


@app.route("/slice", methods=["GET"])
def get_slice():
    mode = request.args.get("mode", "inline")

    with cube_lock:
        if cube is None:
            return jsonify({"error": "No SEG-Y volume loaded."}), 400

        inline_idx = int(request.args.get("inline", 0))
        crossline_idx = int(request.args.get("crossline", 0))
        time_idx = int(request.args.get("time", 0))

        if mode == "inline":
            data = cube[inline_idx, :, :]
        elif mode == "crossline":
            data = cube[:, crossline_idx, :]
        elif mode == "time":
            data = cube[:, :, time_idx]
        else:
            return jsonify({"error": f"Unsupported mode: {mode}"}), 400

    return jsonify({"shape": list(data.shape), "data": data.tolist()})


if __name__ == "__main__":
    print("Starting SEG-Y slice server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000)
