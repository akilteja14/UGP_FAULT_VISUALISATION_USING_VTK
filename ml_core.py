import numpy as np
import segyio
import tensorflow as tf
from tensorflow.keras.models import model_from_json, Model
import os

PATCH_SIZE = 128
OVERLAP = 12

def load_model():
    print("Loading trained model...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_json_path = os.path.join(base_dir, 'model', 'model3.json')
    weights_path = os.path.join(base_dir, 'model', 'pretrained_model.hdf5')

    if not os.path.exists(model_json_path):
        raise FileNotFoundError(f"Model JSON not found at {model_json_path}. Ensure 'model/model3.json' exists.")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}. Place your weights at 'model/pretrained_model.hdf5' or update the path.")

    with open(model_json_path, 'r') as f:
        model_json = f.read()

    with tf.keras.utils.custom_object_scope({'Model': Model}):
        model = model_from_json(model_json)

    model.load_weights(weights_path)
    model.trainable = False
    return model

def create_overlap_mask():
    n, overlap = PATCH_SIZE, OVERLAP
    mask = np.ones((n, n, n), dtype=np.float32)
    edge = np.zeros(overlap, dtype=np.float32)
    sigma = 0.5 / ((overlap / 4) ** 2)
    for k in range(overlap):
        d = k - overlap + 1
        edge[k] = np.exp(-d * d * sigma)
    for k in range(overlap):
        mask[k,:,:] *= edge[k]
        mask[-k-1,:,:] *= edge[k]
        mask[:,k,:] *= edge[k]
        mask[:,-k-1,:] *= edge[k]
        mask[:,:,k] *= edge[k]
        mask[:,:,-k-1] *= edge[k]
    return mask

def process_segy(input_path, output_path, model):
    try:
        print(f"[ML] Reading input file: {input_path}")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        print("[ML] Opening SEGY file...")
        with segyio.open(input_path, "r", ignore_geometry=True) as f:
            n_traces = f.tracecount
            n_samples = len(f.samples)
            print(f"[ML] Input dimensions: {n_traces} traces x {n_samples} samples")
            
            try:
                # Try standard 3D cube reconstruction
                print("[ML] Attempting to load as 3D cube...")
                volume = segyio.tools.cube(f)
                print(f"[ML] Loaded as 3D cube: shape {volume.shape}")
            except (TypeError, ValueError) as e:
                print(f"[ML] 3D cube load failed: {e}")
                # DETECTIVE WORK: Find actual dimensions from headers 
                print("[ML] Extracting dimensions from headers...")
                inlines = f.attributes(segyio.TraceField.INLINE_3D)[:]
                crosslines = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]
                
                n_in = len(np.unique(inlines))
                n_cr = len(np.unique(crosslines))
                print(f"[ML] Detected grid: {n_in} inlines x {n_cr} crosslines")
                
                if n_in * n_cr == n_traces:
                    print("[ML] Reshaping to 3D volume...")
                    volume = f.trace.raw[:].reshape(n_in, n_cr, n_samples)
                    print(f"[ML] Reshaped to: {volume.shape}")
                else:
                    raise ValueError(f"Geometry mismatch: {n_traces} traces vs {n_in}x{n_cr} grid.")

        volume = volume.astype(np.float32)
        m1, m2, m3 = volume.shape
        print(f"[ML] Volume shape: {m1} x {m2} x {m3}")

        # Normalize to 0-255 for the CNN
        print("[ML] Normalizing volume...")
        volume -= volume.min()
        volume /= (volume.max() + 1e-8)
        volume *= 255.0
        print("[ML] Normalization complete")

        n, overlap = PATCH_SIZE, OVERLAP
        c1, c2, c3 = [int(np.ceil((m - overlap) / (n - overlap))) for m in [m1, m2, m3]]
        print(f"[ML] Patch grid: {c1} x {c2} x {c3} = {c1*c2*c3} patches")

        p1, p2, p3 = [(n-overlap)*c + overlap for c in [c1, c2, c3]]
        print(f"[ML] Padded shape: {p1} x {p2} x {p3}")

        # Use disk-backed memmaps to avoid large RAM spikes
        import tempfile, uuid
        tmpdir = tempfile.gettempdir()
        uid = uuid.uuid4().hex
        padded_path = os.path.join(tmpdir, f"padded_{uid}.dat")
        output_path_tmp = os.path.join(tmpdir, f"output_{uid}.dat")
        weight_path_tmp = os.path.join(tmpdir, f"weight_{uid}.dat")

        try:
            padded = np.memmap(padded_path, dtype=np.float32, mode='w+', shape=(p1, p2, p3))
            # copy volume into padded region
            padded[:m1, :m2, :m3] = volume
            output = np.memmap(output_path_tmp, dtype=np.float32, mode='w+', shape=(p1, p2, p3))
            output[:] = 0.0
            weight = np.memmap(weight_path_tmp, dtype=np.float32, mode='w+', shape=(p1, p2, p3))
            weight[:] = 0.0
        except Exception as e:
            # Fall back to in-memory arrays if memmap creation fails
            print(f"[ML] Memmap allocation failed: {e}; falling back to in-memory arrays")
            padded = np.zeros((p1, p2, p3), dtype=np.float32)
            padded[:m1, :m2, :m3] = volume
            output = np.zeros_like(padded)
            weight = np.zeros_like(padded)
        
        print("[ML] Creating overlap mask...")
        mask = create_overlap_mask()
        print("[ML] Mask created")

        print("[ML] Starting patch processing...")
        total_patches = c1 * c2 * c3
        patch_count = 0
        
        for i in range(c1):
            for j in range(c2):
                for k in range(c3):
                    b1, b2, b3 = i*(n-overlap), j*(n-overlap), k*(n-overlap)
                    patch = padded[b1:b1+n, b2:b2+n, b3:b3+n].reshape(1, n, n, n, 1)
                    pred = model.predict(patch, verbose=0)[0, :, :, :, 0]
                    output[b1:b1+n, b2:b2+n, b3:b3+n] += pred * mask
                    weight[b1:b1+n, b2:b2+n, b3:b3+n] += mask
                    
                    patch_count += 1
                    if patch_count % max(1, total_patches // 10) == 0:  # Log every 10%
                        progress = int((patch_count / total_patches) * 100)
                        print(f"[ML] Progress: {patch_count}/{total_patches} ({progress}%)")

        print("[ML] Patch processing complete, finalizing result...")

        # Compute result in chunks to avoid allocating the full array in RAM
        eps = 1e-8
        def iter_result_slices():
            # iterate over the first axis (inlines) to produce (m2, m3) slices
            for i in range(m1):
                out_slice = output[i, :m2, :m3]
                w_slice = weight[i, :m2, :m3]
                with np.errstate(divide='ignore', invalid='ignore'):
                    res_slice = out_slice / (w_slice + eps)
                yield res_slice

        # Determine flat trace count and prepare to write incrementally
        flat_trace_count = m1 * m2
        print(f"[ML] Preparing to write {flat_trace_count} traces incrementally...")

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            print(f"[ML] Output directory ready: {output_dir}")

        print(f"[ML] Writing output file: {output_path}")
        with segyio.open(input_path, "r", ignore_geometry=True) as src:
            spec = segyio.spec()
            spec.samples = src.samples
            spec.format = src.format

            # Write traces incrementally to avoid building the full result in RAM
            spec.tracecount = m1 * m2
            print(f"[ML] Setting spec.tracecount to {spec.tracecount}")
            print("[ML] Creating SEGY file (streaming writes)...")
            with segyio.create(output_path, spec) as dst:
                write_index = 0
                for i, res_slice in enumerate(iter_result_slices()):
                    # res_slice has shape (m2, m3)
                    dst.trace[write_index:write_index + m2] = res_slice
                    write_index += m2
                    if i % max(1, m1 // 10) == 0:
                        print(f"[ML] Written inline {i+1}/{m1}")
                print(f"[ML] Wrote {write_index} traces")

        print(f"[ML] Successfully saved output to: {output_path}")
        if not os.path.exists(output_path):
            raise RuntimeError(f"Output file was not created at {output_path}")

        file_size = os.path.getsize(output_path)
        print(f"[ML] Output file size: {file_size} bytes")
        print("[ML] Process completed successfully!")

        # Clean up temporary memmap files if they exist
        try:
            for path in (padded_path, output_path_tmp, weight_path_tmp):
                if 'path' in locals() and path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
        except Exception:
            pass

        return True
        
    except Exception as err:
        print(f"[ML] ERROR: {type(err).__name__}: {err}")
        import traceback
        traceback.print_exc()
        raise