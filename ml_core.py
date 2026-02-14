import numpy as np
import segyio
import tensorflow as tf
from tensorflow.keras.models import model_from_json, Model
import os

PATCH_SIZE = 128
OVERLAP = 12

def load_model():
    print("Loading trained model...")
    with open('model/model3.json', 'r') as f:
        model_json = f.read()

    with tf.keras.utils.custom_object_scope({'Model': Model}):
        model = model_from_json(model_json)
    
    model.load_weights("model/pretrained_model.hdf5")
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
        result = (output / (weight + 1e-8))[:m1, :m2, :m3]
        print(f"[ML] Result shape: {result.shape}")
        print(f"[ML] Result data range: {result.min():.2f} - {result.max():.2f}")

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
            
            # Calculate total number of traces
            flat_result = result.reshape(-1, result.shape[-1])
            spec.tracecount = len(flat_result)
            print(f"[ML] Setting spec.tracecount to {spec.tracecount}")
            
            print("[ML] Creating SEGY file...")
            with segyio.create(output_path, spec) as dst:
                # Write traces back in the same shape as input
                dst.trace[:] = flat_result
                print(f"[ML] Wrote {len(flat_result)} traces")

        print(f"[ML] Successfully saved output to: {output_path}")
        if not os.path.exists(output_path):
            raise RuntimeError(f"Output file was not created at {output_path}")
        
        file_size = os.path.getsize(output_path)
        print(f"[ML] Output file size: {file_size} bytes")
        print("[ML] Process completed successfully!")
        return True
        
    except Exception as err:
        print(f"[ML] ERROR: {type(err).__name__}: {err}")
        import traceback
        traceback.print_exc()
        raise