import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import segyio
import tensorflow as tf
from tensorflow.keras.models import model_from_json, Model
import tempfile
import uuid
import time

# ---------------- SETTINGS ---------------- #
PATCH_SIZE = 128
OVERLAP = 12
BATCH_SIZE = 1   # Increase to 16/32 if GPU allows


# ---------------- MODEL ---------------- #
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPUs detected:", gpus)
    except Exception as e:
        print("Could not set memory growth:", e)
else:
    print("Running on CPU")



def load_model():
    print("[ML] Loading model...")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "model", "model3.json")
    weights_path = os.path.join(base_dir, "model", "pretrained_model.hdf5")

    with open(json_path, "r") as f:
        model_json = f.read()

    with tf.keras.utils.custom_object_scope({"Model": Model}):
        model = model_from_json(model_json)

    model.load_weights(weights_path)
    model.trainable = False

    print("✅ Model loaded")
    return model


# ---------------- MASK ---------------- #
def create_overlap_mask():
    n, os_val = PATCH_SIZE, OVERLAP

    mask = np.ones((n, n, n), dtype=np.float32)
    edge = np.zeros(os_val, dtype=np.float32)

    sigma = 0.5 / ((os_val / 4) ** 2)

    for k in range(os_val):
        d = k - os_val + 1
        edge[k] = np.exp(-d * d * sigma)

    for k in range(os_val):
        mask[k,:,:] *= edge[k]
        mask[-k-1,:,:] *= edge[k]
        mask[:,k,:] *= edge[k]
        mask[:,-k-1,:] *= edge[k]
        mask[:,:,k] *= edge[k]
        mask[:,:,-k-1] *= edge[k]

    return mask


# ---------------- BATCH INFERENCE ---------------- #
def run_batch(model, patches, coords, output, weight, mask):
    batch = np.array(patches, dtype=np.float32)[..., np.newaxis]

    preds = model.predict(batch, verbose=0)

    for idx, (b1, b2, b3) in enumerate(coords):
        pred = preds[idx, :, :, :, 0]

        output[b1:b1+PATCH_SIZE,
               b2:b2+PATCH_SIZE,
               b3:b3+PATCH_SIZE] += pred * mask

        weight[b1:b1+PATCH_SIZE,
               b2:b2+PATCH_SIZE,
               b3:b3+PATCH_SIZE] += mask


# ---------------- MAIN ---------------- #
def process_segy(input_path, output_path, model, progress_callback=None):

    def report_progress(stage, message, percent=None, extra=None):
        if progress_callback is not None:
            payload = {"stage": stage, "message": message}
            if percent is not None:
                payload["percent"] = percent
            if extra:
                payload.update(extra)
            progress_callback(payload)

    print("[ML] Reading SEG-Y...")
    report_progress("reading", "Reading SEG-Y...", 5)

    with segyio.open(input_path, "r", ignore_geometry=False) as src:
        m1 = len(src.ilines)
        m2 = len(src.xlines)
        m3 = len(src.samples)
        spec = segyio.tools.metadata(src)

    # ---------------- PATCH SETUP ---------------- #
    n, os_val = PATCH_SIZE, OVERLAP

    c1 = int(np.ceil((m1 - os_val) / (n - os_val)))
    c2 = int(np.ceil((m2 - os_val) / (n - os_val)))
    c3 = int(np.ceil((m3 - os_val) / (n - os_val)))

    p1 = (n - os_val) * c1 + os_val
    p2 = (n - os_val) * c2 + os_val
    p3 = (n - os_val) * c3 + os_val

    total_patches = c1 * c2 * c3

    print(f"[ML] Total patches: {total_patches}")
    report_progress(
        "preparing",
        f"Prepared {total_patches} inference patches.",
        12,
        {"total_patches": total_patches, "volume_shape": [int(m1), int(m2), int(m3)]},
    )

    # ---------------- MEMMAP ---------------- #
    uid = uuid.uuid4().hex
    padded_path = os.path.join(tempfile.gettempdir(), f"pad_{uid}.dat")
    out_path = os.path.join(tempfile.gettempdir(), f"out_{uid}.dat")
    wt_path = os.path.join(tempfile.gettempdir(), f"wt_{uid}.dat")

    padded = np.memmap(padded_path, dtype="float32", mode="w+", shape=(p1,p2,p3))
    
    # Efficiently load data inline-by-inline directly into the memmap
    # to avoid loading the entire cube into RAM at once.
    with segyio.open(input_path, "r", ignore_geometry=False) as src:
        for i, iline_no in enumerate(src.ilines):
            padded[i, :m2, :m3] = src.iline[iline_no]

    output = np.memmap(out_path, dtype="float32", mode="w+", shape=(p1,p2,p3))
    weight = np.memmap(wt_path, dtype="float32", mode="w+", shape=(p1,p2,p3))

    output[:] = 0
    weight[:] = 0

    mask = create_overlap_mask()

    # ---------------- FAST INFERENCE ---------------- #
    print("[ML] Starting FAST GPU inference...")
    
    completed_patches = 0
    progress_step = max(1, total_patches // 20)
    start_time = time.time()
    report_progress("inference", "Starting inference...", 15, {"total_patches": total_patches, "completed_patches": 0})

    patches, coords = [], []

    for i in range(c1):
        for j in range(c2):
            for k in range(c3):

                b1 = i * (n - os_val)
                b2 = j * (n - os_val)
                b3 = k * (n - os_val)

                patch = padded[b1:b1+n, b2:b2+n, b3:b3+n]

                # FAST NORMALIZATION
                p_min = patch.min()
                p_max = patch.max()
                patch = (patch - p_min) / (p_max - p_min + 1e-8) * 255.0

                patches.append(patch)
                coords.append((b1, b2, b3))

                #  BATCH EXECUTION
                if len(patches) == BATCH_SIZE:
                    run_batch(model, patches, coords, output, weight, mask)
                    patches, coords = [], []

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

    # leftover
    if patches:
        run_batch(model, patches, coords, output, weight, mask)

    # ---------------- FINALIZE ---------------- #
    print("[ML] Stitching...")
    report_progress("stitching", "Stitching patches...", 90)
    result = output / (weight + 1e-8)
    result = result[:m1, :m2, :m3]

    print("[ML] Writing SEG-Y...")
    report_progress("writing", "Finalizing and writing output SEG-Y...", 92)

    with segyio.open(input_path, "r") as src:
        with segyio.create(output_path, spec) as dst:
            dst.text[0] = src.text[0]
            dst.bin = src.bin

            for i in range(m1):
                for j in range(m2):
                    idx = i * m2 + j
                    dst.trace[idx] = result[i, j, :]
                    dst.header[idx] = src.header[idx]

    print("✅ Done in {:.2f}s".format(time.time() - start_time))
    report_progress("complete", "Process Complete.", 100, {"output_path": output_path})