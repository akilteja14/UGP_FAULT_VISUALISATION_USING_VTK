import segyio
import numpy as np
import vtk
from vtk.util import numpy_support
import base64

# Note: Adjust this import to match wherever your actual ML logic lives
from ml_core import load_model, process_segy

# --- ML Model Management ---
_model_cache = None

def get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = load_model()
    return _model_cache

def run_ml_extraction(input_path, output_path):
    """Wrapper to run the ML model."""
    model = get_model()
    process_segy(input_path, output_path, model)
    return True

# --- Data Extraction ---
def build_numpy_cube(filename):
    """Reads SEGY and builds the numpy array."""
    with segyio.open(filename, ignore_geometry=True) as f:
        try:
            inlines = f.attributes(segyio.TraceField.INLINE_3D)[:]
            crosslines = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]
        except Exception:
            inlines = f.attributes(181)[:]
            crosslines = f.attributes(185)[:]

        samples = f.samples
        unique_inlines = np.unique(inlines)
        unique_crosslines = np.unique(crosslines)

        n_inline = len(unique_inlines)
        n_crossline = len(unique_crosslines)
        n_samples = len(samples)

        inline_index = {v: i for i, v in enumerate(unique_inlines)}
        crossline_index = {v: i for i, v in enumerate(unique_crosslines)}

        cube = np.zeros((n_inline, n_crossline, n_samples), dtype=np.float32)

        for trace_id in range(f.tracecount):
            il = inlines[trace_id]
            xl = crosslines[trace_id]
            if il in inline_index and xl in crossline_index:
                i = inline_index[il]
                j = crossline_index[xl]
                cube[i, j, :] = f.trace[trace_id]

    return cube, unique_inlines, unique_crosslines, samples


# --- VTK Visualization ---
def create_vtk_mapper(cube):
    """Takes numpy array and builds VTK objects."""
    vtk_data = numpy_support.numpy_to_vtk(
        cube.ravel(order="F"),
        deep=True,
        array_type=vtk.VTK_FLOAT
    )

    image = vtk.vtkImageData()
    nx, ny, nz = cube.shape
    image.SetDimensions(nx, ny, nz)
    image.GetPointData().SetScalars(vtk_data)

    amp_abs = np.abs(cube)
    clip = np.percentile(amp_abs, 99) if amp_abs.size > 0 else 1.0

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(-clip, clip)

    for i in range(256):
        x = i / 255.0
        if x < 0.5:
            r, g, b = (x*2, x*2, 1.0)
        else:
            r, g, b = (1.0, 2*(1-x), 2*(1-x))
        lut.SetTableValue(i, r, g, b, 1.0)
    lut.Build()

    color_mapper = vtk.vtkImageMapToColors()
    color_mapper.SetLookupTable(lut)
    color_mapper.SetInputData(image)
    color_mapper.Update()

    return color_mapper

def render_vtk_slice(color_mapper, shape, slices_config):
    """
    Renders the 3D scene off-screen and returns a base64 JPEG string.
    slices_config is a dict: {'iline': (True/False, index), ...}
    """
    if color_mapper is None:
        return None

    renderer = vtk.vtkRenderer()
    render_win = vtk.vtkRenderWindow()
    render_win.SetOffScreenRendering(1)
    render_win.AddRenderer(renderer)

    def add_slice(axis, index):
        actor = vtk.vtkImageActor()
        actor.GetMapper().SetInputConnection(color_mapper.GetOutputPort())

        if axis == 'iline':
            actor.SetDisplayExtent(index, index, 0, shape[1]-1, 0, shape[2]-1)
        elif axis == 'xline':
            actor.SetDisplayExtent(0, shape[0]-1, index, index, 0, shape[2]-1)
        elif axis == 'time':
            actor.SetDisplayExtent(0, shape[0]-1, 0, shape[1]-1, index, index)

        renderer.AddActor(actor)

    for axis, (is_active, index) in slices_config.items():
        if is_active:
            add_slice(axis, int(index))

    renderer.SetBackground(0.1, 0.1, 0.1)
    renderer.ResetCamera()
    render_win.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(render_win)
    w2i.Update()

    writer = vtk.vtkJPEGWriter()
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.WriteToMemoryOn()
    writer.Write()

    img_b64 = base64.b64encode(writer.GetResult().ToArray()).decode()
    return img_b64