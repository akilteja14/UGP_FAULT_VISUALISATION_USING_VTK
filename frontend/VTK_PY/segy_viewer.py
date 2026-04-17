import numpy as np
import vtk
from vtk.util import numpy_support


def build_vtk_image(cube):
    vtk_data = numpy_support.numpy_to_vtk(
        cube.ravel(order="F"),
        deep=True,
        array_type=vtk.VTK_FLOAT,
    )

    image = vtk.vtkImageData()
    image.SetDimensions(cube.shape)
    image.GetPointData().SetScalars(vtk_data)
    return image


def build_mapper(image, amplitudes):
    amp_abs = np.abs(amplitudes)
    clip = np.percentile(amp_abs, 99) if amp_abs.size else 1.0
    if clip <= 0:
        clip = 1.0

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(-clip, clip)
    lut.Build()

    for i in range(256):
        x = i / 255.0
        if x < 0.5:
            r = g = x * 2
            b = 1.0
        else:
            r = 1.0
            g = b = 2 * (1 - x)
        lut.SetTableValue(i, r, g, b, 1.0)

    mapper = vtk.vtkImageMapToColors()
    mapper.SetLookupTable(lut)
    mapper.SetInputData(image)
    mapper.Update()
    return mapper


def create_actors(mapper, inline_idx, crossline_idx, time_idx, shape):
    inline_actor = vtk.vtkImageActor()
    inline_actor.GetMapper().SetInputConnection(mapper.GetOutputPort())
    inline_actor.SetDisplayExtent(inline_idx, inline_idx, 0, shape[1] - 1, 0, shape[2] - 1)

    cross_actor = vtk.vtkImageActor()
    cross_actor.GetMapper().SetInputConnection(mapper.GetOutputPort())
    cross_actor.SetDisplayExtent(0, shape[0] - 1, crossline_idx, crossline_idx, 0, shape[2] - 1)

    time_actor = vtk.vtkImageActor()
    time_actor.GetMapper().SetInputConnection(mapper.GetOutputPort())
    time_actor.SetDisplayExtent(0, shape[0] - 1, 0, shape[1] - 1, time_idx, time_idx)

    return inline_actor, cross_actor, time_actor
