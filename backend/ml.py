import numpy as np
import segyio
import tensorflow as tf
from tensorflow.keras.models import model_from_json, Model
import os
import tempfile
import uuid
import time

# ---------------- SETTINGS ---------------- #
PATCH_SIZE = 128
OVERLAP = 12
BATCH_SIZE = 8   # Increase to 16/32 if GPU allows

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU detected:", gpus[0])
    except Exception as e:
        print("Could not set memory growth:", e)

# ---------------- MODEL ---------------- #
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

    def report(stage, msg, pct=None):
        if progress_callback:
            progress_callback({"stage": stage, "message": msg, "percent": pct})

    print("[ML] Reading SEG-Y...")
    report("reading", "Reading SEG-Y...", 5)

    with segyio.open(input_path, "r", ignore_geometry=False) as src:
        volume = segyio.tools.cube(src)
        spec = segyio.tools.metadata(src)

    volume = volume.astype(np.float32)
    m1, m2, m3 = volume.shape

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

    # ---------------- MEMMAP ---------------- #
    uid = uuid.uuid4().hex
    padded_path = os.path.join(tempfile.gettempdir(), f"pad_{uid}.dat")
    out_path = os.path.join(tempfile.gettempdir(), f"out_{uid}.dat")
    wt_path = os.path.join(tempfile.gettempdir(), f"wt_{uid}.dat")

    padded = np.memmap(padded_path, dtype="float32", mode="w+", shape=(p1,p2,p3))
    padded[:m1,:m2,:m3] = volume

    output = np.memmap(out_path, dtype="float32", mode="w+", shape=(p1,p2,p3))
    weight = np.memmap(wt_path, dtype="float32", mode="w+", shape=(p1,p2,p3))

    output[:] = 0
    weight[:] = 0

    mask = create_overlap_mask()

    # ---------------- FAST INFERENCE ---------------- #
    print("[ML] Starting FAST GPU inference...")
    report("inference", "Running inference...", 15)

    patches, coords = [], []
    completed = 0
    start = time.time()

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

                completed += 1

                if completed % 100 == 0:
                    elapsed = time.time() - start
                    print(f"{completed}/{total_patches} | {elapsed:.1f}s")

    # leftover
    if patches:
        run_batch(model, patches, coords, output, weight, mask)

    # ---------------- FINALIZE ---------------- #
    print("[ML] Stitching...")
    result = output / (weight + 1e-8)
    result = result[:m1, :m2, :m3]

    print("[ML] Writing SEG-Y...")

    with segyio.open(input_path, "r") as src:
        with segyio.create(output_path, spec) as dst:
            dst.text[0] = src.text[0]
            dst.bin = src.bin

            for i in range(m1):
                for j in range(m2):
                    idx = i * m2 + j
                    dst.trace[idx] = result[i, j, :]
                    dst.header[idx] = src.header[idx]

    print("✅ Done in {:.2f}s".format(time.time() - start))