import numpy as np
import vtk
from vtk.util import numpy_support

#---- WHOLE VTK ARCHITECTURE ----#
# Data → Representation → Mapping → Actors → Renderer → Window
# Numpy Cube -> vtkImageData -> vtkImageMapToColors -> vtkImageActors -> Renderer
# -> RenderWindow (view on screen or web).

# Converts a NumPy 3D seismic volume into vtkImageData
# so that VTK can render it as image slices.
def build_vtk_image(cube):
    #we are flattening the 3D numpy array the volume which comes from
    #build_sparse_cube in "F" fortran order(column major order) because vtk
    #expects data in column major order,
    #deep = true creates a safe independent copy of data and,
    #array_type is float
    vtk_data = numpy_support.numpy_to_vtk(
        cube.ravel(order="F"),
        deep=True,
        array_type=vtk.VTK_FLOAT,
    )

    #creates an empty 3D image
    image = vtk.vtkImageData()
    #sets the 3D image the dimensions of our cube
    image.SetDimensions(cube.shape)
    #sets the amplitude of every point from the above vtk_data
    image.GetPointData().SetScalars(vtk_data)

    #returns the image
    return image


def build_mapper(image, amplitudes):
    #we use the amplitudes array created in the update_slices function here
    #using absolute value value of amplitude as magnitude is important 
    amp_abs = np.abs(amplitudes)
    #taking 99th percentile to avoid outliers to change coloring scheme
    #if amp_abs.size is zero then taking clip = 1.0
    clip = np.percentile(amp_abs, 99) if amp_abs.size else 1.0
    if clip <= 0:
        clip = 1.0

    #creating a lookup table of size 256
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)

    #setting range of lookup table and building it
    lut.SetRange(-clip, clip)
    lut.Build()

    for i in range(256):
        x = i / 255.0
        #first half blue -> white transition
        if x < 0.5:
            r = g = x * 2
            b = 1.0
        #second half white -> red transition
        else:
            r = 1.0
            g = b = 2 * (1 - x)

        #setting lookup table color scheme
        lut.SetTableValue(i, r, g, b, 1.0)

    #creating a mapper
    mapper = vtk.vtkImageMapToColors()
    #telling it to use the above LUT table
    mapper.SetLookupTable(lut)
    #setting input data as image (i.e vtk scalars)
    mapper.SetInputData(image)
    #apply the lookup table to the image
    mapper.Update()

    #return the mapper
    return mapper


def create_actors(mapper, inline_idx, crossline_idx, time_idx, shape):
    #creating image actors for inline, crossline, time slice
    inline_actor = vtk.vtkImageActor()
    #output of mapper to input of actor as per vtk architecture or pipeline
    inline_actor.GetMapper().SetInputConnection(mapper.GetOutputPort())
    #for inline view all crosslines and times for fixed inline_idx
    inline_actor.SetDisplayExtent(inline_idx, inline_idx, 0, shape[1] - 1, 0, shape[2] - 1)

    #similarily
    cross_actor = vtk.vtkImageActor()
    cross_actor.GetMapper().SetInputConnection(mapper.GetOutputPort())
    cross_actor.SetDisplayExtent(0, shape[0] - 1, crossline_idx, crossline_idx, 0, shape[2] - 1)

    #similarily
    time_actor = vtk.vtkImageActor()
    time_actor.GetMapper().SetInputConnection(mapper.GetOutputPort())
    time_actor.SetDisplayExtent(0, shape[0] - 1, 0, shape[1] - 1, time_idx, time_idx)

    #return the actors to update_slice function...
    return inline_actor, cross_actor, time_actor
