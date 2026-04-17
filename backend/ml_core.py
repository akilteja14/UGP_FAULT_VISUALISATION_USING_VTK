import numpy as np
import segyio
import tensorflow as tf
from tensorflow.keras.models import model_from_json, Model
import os
import tempfile
import uuid
import time

PATCH_SIZE = 128
OVERLAP = 12

def load_model():
    print("[ML] Loading trained model...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_json_path = os.path.join(base_dir, 'model', 'model3.json')
    weights_path = os.path.join(base_dir, 'model', 'pretrained_model.hdf5')

    with open(model_json_path, 'r') as f:
        model_json = f.read()

    # Using custom_object_scope to handle potential Keras functional API issues
    with tf.keras.utils.custom_object_scope({'Model': Model}):
        model = model_from_json(model_json)

    model.load_weights(weights_path)
    model.trainable = False
    return model

def create_overlap_mask():
    n, os_val = PATCH_SIZE, OVERLAP
    mask = np.ones((n, n, n), dtype=np.float32)
    edge = np.zeros(os_val, dtype=np.float32)
    sigma = 0.5 / ((os_val / 4) ** 2)
    for k in range(os_val):
        d = k - os_val + 1
        edge[k] = np.exp(-d * d * sigma)
    for k in range(os_val):
        mask[k,:,:] *= edge[k]; mask[-k-1,:,:] *= edge[k]
        mask[:,k,:] *= edge[k]; mask[:,-k-1,:] *= edge[k]
        mask[:,:,k] *= edge[k]; mask[:,:,-k-1] *= edge[k]
    return mask

def process_segy(input_path, output_path, model, progress_callback=None):
    padded_path = None
    out_tmp_path = None
    wt_tmp_path = None
    padded = None
    output = None
    weight = None

    def report_progress(stage, message, percent=None, extra=None):
        if progress_callback is not None:
            payload = {"stage": stage, "message": message}
            if percent is not None:
                payload["percent"] = percent
            if extra:
                payload.update(extra)
            progress_callback(payload)

    try:
        print(f"[ML] Reading: {input_path}")
        report_progress("reading", "Reading SEG-Y volume...", 2)
        
        # 1. ATTEMPT STRUCTURED READ (Script 2 style)
        try:
            with segyio.open(input_path, "r", ignore_geometry=False) as src:
                volume = segyio.tools.cube(src)
                spec = segyio.tools.metadata(src) 
                print("[ML] Geometry loaded successfully using segyio engine.")
                report_progress("reading", "Geometry loaded successfully.", 8)
        except Exception as e:
            print(f"[ML] Standard geometry failed ({e}). Falling back to detective work...")
            # Fallback to Script 1 "Detective" logic if geometry is non-standard
            with segyio.open(input_path, "r", ignore_geometry=True) as src:
                # (Simplified detective work here for brevity, assuming cube-like structure)
                volume = src.trace.raw[:].reshape(-1, len(src.samples)) 
                # This would need the full reshape logic from Script 1 if headers are truly broken
                raise RuntimeError("Non-standard geometry detected. Please ensure file has valid INLINE/CROSSLINE headers.")

        m1, m2, m3 = volume.shape
        volume = volume.astype(np.float32)

        # 2. PATCHING SETUP
        n, os_val = PATCH_SIZE, OVERLAP
        c1, c2, c3 = [int(np.ceil((m - os_val) / (n - os_val))) for m in [m1, m2, m3]]
        p1, p2, p3 = [(n - os_val) * c + os_val for c in [c1, c2, c3]]
        total_patches = c1 * c2 * c3
        report_progress(
            "preparing",
            f"Prepared {total_patches} inference patches.",
            12,
            {"total_patches": total_patches, "volume_shape": [int(m1), int(m2), int(m3)]},
        )

        # Use memmaps for large volumes
        uid = uuid.uuid4().hex
        padded_path = os.path.join(tempfile.gettempdir(), f"pad_{uid}.dat")
        out_tmp_path = os.path.join(tempfile.gettempdir(), f"out_{uid}.dat")
        wt_tmp_path = os.path.join(tempfile.gettempdir(), f"wt_{uid}.dat")

        padded = np.memmap(padded_path, dtype='float32', mode='w+', shape=(p1, p2, p3))
        padded[:m1, :m2, :m3] = volume
        output = np.memmap(out_tmp_path, dtype='float32', mode='w+', shape=(p1, p2, p3))
        weight = np.memmap(wt_tmp_path, dtype='float32', mode='w+', shape=(p1, p2, p3))
        output[:] = 0.0; weight[:] = 0.0

        mask = create_overlap_mask()

        # 3. INFERENCE LOOP with PER-PATCH NORMALIZATION
        print("[ML] Starting Inference...")
        completed_patches = 0
        progress_step = max(1, total_patches // 20)
        start_time = time.time()
        report_progress("inference", "Starting inference...", 15, {"total_patches": total_patches, "completed_patches": 0})

        for i in range(c1):
            for j in range(c2):
                for k in range(c3):
                    b1, b2, b3 = i*(n-os_val), j*(n-os_val), k*(n-os_val)
                    patch = padded[b1:b1+n, b2:b2+n, b3:b3+n].copy()

                    # PER-PATCH NORMALIZATION (From Script 2)
                    p_min = patch.min()
                    p_max = patch.max()
                    patch = (patch - p_min) / (p_max - p_min + 1e-8) * 255.0

                    patch_input = patch.reshape(1, n, n, n, 1)
                    pred = model.predict(patch_input, verbose=0)[0, :, :, :, 0]

                    output[b1:b1+n, b2:b2+n, b3:b3+n] += pred * mask
                    weight[b1:b1+n, b2:b2+n, b3:b3+n] += mask

                    completed_patches += 1
                    if completed_patches == 1 or completed_patches % progress_step == 0 or completed_patches == total_patches:
                        elapsed = time.time() - start_time
                        rate = completed_patches / elapsed if elapsed > 0 else 0.0
                        remaining = total_patches - completed_patches
                        eta_seconds = remaining / rate if rate > 0 else 0.0
                        percent = (completed_patches / total_patches) * 100
                        overall_percent = 15 + percent * 0.75
                        print(
                            f"[ML] Progress: {completed_patches}/{total_patches} patches "
                            f"({percent:.1f}%) | Elapsed: {elapsed:.1f}s | ETA: {eta_seconds:.1f}s"
                        )
                        report_progress(
                            "inference",
                            f"Inference {completed_patches}/{total_patches} patches ({percent:.1f}%)",
                            overall_percent,
                            {
                                "completed_patches": completed_patches,
                                "total_patches": total_patches,
                                "elapsed_seconds": round(elapsed, 1),
                                "eta_seconds": round(eta_seconds, 1),
                            },
                        )

        # 4. WRITE RESULT WITH FULL HEADERS (From Script 2)
        print(f"[ML] Finalizing and writing to {output_path}")
        report_progress("writing", "Finalizing and writing output SEG-Y...", 92)
        with segyio.open(input_path, "r") as src:
            # spec was captured during the read phase
            with segyio.create(output_path, spec) as dst:
                # Copy global headers
                dst.text[0] = src.text[0]
                dst.bin = src.bin

                # Write traces and copy trace headers
                for i, il in enumerate(spec.ilines):
                    for j, xl in enumerate(spec.xlines):
                        # Calculate global trace index
                        # Note: This assumes standard sorting.
                        trace_idx = i * len(spec.xlines) + j
                        
                        # Blend result for this trace
                        res_trace = output[i, j, :m3] / (weight[i, j, :m3] + 1e-8)
                        
                        dst.trace[trace_idx] = res_trace
                        dst.header[trace_idx] = src.header[trace_idx]

        print("[ML] Process Complete.")
        report_progress("complete", "Process Complete.", 100, {"output_path": output_path})
        return True

    finally:
        # Flush and release memmap handles before removing temp files on Windows.
        for arr in [padded, output, weight]:
            if arr is not None:
                arr.flush()
        del padded
        del output
        del weight

        # Cleanup temp files
        for f in [padded_path, out_tmp_path, wt_tmp_path]:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except PermissionError:
                    print(f"[ML] Warning: could not delete temp file yet: {f}")
