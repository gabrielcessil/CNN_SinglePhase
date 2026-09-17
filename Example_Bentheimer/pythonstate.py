# state file generated using paraview version 5.13.3
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 13

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1161, 776]
renderView1.AxesGrid = 'Grid Axes 3D Actor'
renderView1.CenterOfRotation = [63.5, 63.5, 119.0]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-225.89365899748728, 296.884236267454, 320.9082364712276]
renderView1.CameraFocalPoint = [39.09199443076134, 62.232744818341544, 92.03952964247274]
renderView1.CameraViewUp = [0.3827641710831522, 0.829313355412831, -0.40710066060980404]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 90.15819430312477
renderView1.LegendGrid = 'Legend Grid Actor'
renderView1.PolarGrid = 'Polar Grid Actor'
renderView1.UseColorPaletteForBackground = 0
renderView1.Background = [1.0, 1.0, 1.0]
renderView1.EnableRayTracing = 1
renderView1.BackEnd = 'OSPRay pathtracer'
renderView1.Shadows = 1
renderView1.AmbientSamples = 6
renderView1.SamplesPerPixel = 4
renderView1.ProgressivePasses = 3
renderView1.Backgroundmode = 'Backplate'
renderView1.EnvironmentalBG = [0.6, 0.5803921568627451, 0.6705882352941176]
renderView1.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1161, 776)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'Image Reader'
geometry_paddedraw = ImageReader(registrationName='geometry_padded.raw', FileNames=['/home/gabriel/remote/hal/dissertacao/CNN_SinglePhase/Example_Bentheimer/geometry_padded.raw'])
geometry_paddedraw.DataScalarType = 'unsigned char'
geometry_paddedraw.DataByteOrder = 'LittleEndian'
geometry_paddedraw.DataExtent = [0, 127, 0, 127, 0, 127]

# create a new 'Image Reader'
domainraw = ImageReader(registrationName='domain.raw', FileNames=['/home/gabriel/remote/hal/dissertacao/CNN_SinglePhase/Example_Bentheimer/domain.raw'])
domainraw.DataScalarType = 'unsigned char'
domainraw.DataExtent = [0, 119, 0, 119, 0, 119]

# create a new 'Threshold'
threshold3 = Threshold(registrationName='Threshold3', Input=domainraw)
threshold3.Scalars = ['POINTS', 'ImageFile']
threshold3.UpperThreshold = 1.0
threshold3.ThresholdMethod = 'Below Lower Threshold'

# create a new 'Clip'
clip1 = Clip(registrationName='Clip1', Input=domainraw)
clip1.ClipType = 'Plane'
clip1.HyperTreeGridClipper = 'Plane'
clip1.Scalars = ['POINTS', 'ImageFile']
clip1.Value = 0.5

# init the 'Plane' selected for 'ClipType'
clip1.ClipType.Origin = [59.5, 59.5, 111.0]
clip1.ClipType.Normal = [0.0, 0.0, -1.0]

# init the 'Plane' selected for 'HyperTreeGridClipper'
clip1.HyperTreeGridClipper.Origin = [59.5, 59.5, 59.5]

# create a new 'Threshold'
threshold2 = Threshold(registrationName='Threshold2', Input=clip1)
threshold2.Scalars = ['POINTS', 'ImageFile']
threshold2.UpperThreshold = 1.0
threshold2.ThresholdMethod = 'Below Lower Threshold'

# create a new 'Clip'
clip3 = Clip(registrationName='Clip3', Input=geometry_paddedraw)
clip3.ClipType = 'Plane'
clip3.HyperTreeGridClipper = 'Plane'
clip3.Scalars = ['POINTS', 'ImageFile']
clip3.Value = 0.5

# init the 'Plane' selected for 'ClipType'
clip3.ClipType.Origin = [63.5, 119.0, 63.5]
clip3.ClipType.Normal = [0.0, -1.0, 0.0]

# init the 'Plane' selected for 'HyperTreeGridClipper'
clip3.HyperTreeGridClipper.Origin = [63.5, 63.5, 63.5]

# create a new 'Clip'
clip2 = Clip(registrationName='Clip2', Input=geometry_paddedraw)
clip2.ClipType = 'Plane'
clip2.HyperTreeGridClipper = 'Plane'
clip2.Scalars = ['POINTS', 'ImageFile']
clip2.Value = 0.5

# init the 'Plane' selected for 'ClipType'
clip2.ClipType.Origin = [59.5, 59.5, 119.0]
clip2.ClipType.Normal = [0.0, 0.0, -1.0]

# init the 'Plane' selected for 'HyperTreeGridClipper'
clip2.HyperTreeGridClipper.Origin = [59.5, 59.5, 59.5]

# create a new 'Threshold'
threshold1 = Threshold(registrationName='Threshold1', Input=clip2)
threshold1.Scalars = ['POINTS', 'ImageFile']
threshold1.UpperThreshold = 1.0
threshold1.ThresholdMethod = 'Below Lower Threshold'

# create a new 'Clip'
clip4 = Clip(registrationName='Clip4', Input=clip3)
clip4.ClipType = 'Plane'
clip4.HyperTreeGridClipper = 'Plane'
clip4.Scalars = ['POINTS', 'ImageFile']

# init the 'Plane' selected for 'ClipType'
clip4.ClipType.Origin = [63.5, 123.0, 119.0]
clip4.ClipType.Normal = [0.0, 0.0, 1.0]

# init the 'Plane' selected for 'HyperTreeGridClipper'
clip4.HyperTreeGridClipper.Origin = [63.5, 123.0, 63.5]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from geometry_paddedraw
geometry_paddedrawDisplay = Show(geometry_paddedraw, renderView1, 'UniformGridRepresentation')

# get 2D transfer function for 'ImageFile'
imageFileTF2D = GetTransferFunction2D('ImageFile')
imageFileTF2D.ScalarRangeInitialized = 1

# get color transfer function/color map for 'ImageFile'
imageFileLUT = GetColorTransferFunction('ImageFile')
imageFileLUT.TransferFunction2D = imageFileTF2D
imageFileLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'ImageFile'
imageFilePWF = GetOpacityTransferFunction('ImageFile')
imageFilePWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
geometry_paddedrawDisplay.Representation = 'Surface'
geometry_paddedrawDisplay.ColorArrayName = ['POINTS', 'ImageFile']
geometry_paddedrawDisplay.LookupTable = imageFileLUT
geometry_paddedrawDisplay.Opacity = 0.68
geometry_paddedrawDisplay.SelectNormalArray = 'None'
geometry_paddedrawDisplay.SelectTangentArray = 'None'
geometry_paddedrawDisplay.SelectTCoordArray = 'None'
geometry_paddedrawDisplay.TextureTransform = 'Transform2'
geometry_paddedrawDisplay.OSPRayScaleArray = 'ImageFile'
geometry_paddedrawDisplay.OSPRayScaleFunction = 'Piecewise Function'
geometry_paddedrawDisplay.Assembly = ''
geometry_paddedrawDisplay.SelectedBlockSelectors = ['']
geometry_paddedrawDisplay.SelectOrientationVectors = 'None'
geometry_paddedrawDisplay.ScaleFactor = 12.700000000000001
geometry_paddedrawDisplay.SelectScaleArray = 'ImageFile'
geometry_paddedrawDisplay.GlyphType = 'Arrow'
geometry_paddedrawDisplay.GlyphTableIndexArray = 'ImageFile'
geometry_paddedrawDisplay.GaussianRadius = 0.635
geometry_paddedrawDisplay.SetScaleArray = ['POINTS', 'ImageFile']
geometry_paddedrawDisplay.ScaleTransferFunction = 'Piecewise Function'
geometry_paddedrawDisplay.OpacityArray = ['POINTS', 'ImageFile']
geometry_paddedrawDisplay.OpacityTransferFunction = 'Piecewise Function'
geometry_paddedrawDisplay.DataAxesGrid = 'Grid Axes Representation'
geometry_paddedrawDisplay.PolarAxes = 'Polar Axes Representation'
geometry_paddedrawDisplay.ScalarOpacityUnitDistance = 1.732050807568877
geometry_paddedrawDisplay.ScalarOpacityFunction = imageFilePWF
geometry_paddedrawDisplay.TransferFunction2D = imageFileTF2D
geometry_paddedrawDisplay.OpacityArrayName = ['POINTS', 'ImageFile']
geometry_paddedrawDisplay.ColorArray2Name = ['POINTS', 'ImageFile']
geometry_paddedrawDisplay.IsosurfaceValues = [0.5]
geometry_paddedrawDisplay.SliceFunction = 'Plane'
geometry_paddedrawDisplay.Slice = 63
geometry_paddedrawDisplay.SelectInputVectors = [None, '']
geometry_paddedrawDisplay.WriteLog = ''

# init the 'Plane' selected for 'SliceFunction'
geometry_paddedrawDisplay.SliceFunction.Origin = [63.5, 63.5, 63.5]

# show data from domainraw
domainrawDisplay = Show(domainraw, renderView1, 'UniformGridRepresentation')

# trace defaults for the display properties.
domainrawDisplay.Representation = 'Surface'
domainrawDisplay.ColorArrayName = ['POINTS', 'ImageFile']
domainrawDisplay.LookupTable = imageFileLUT
domainrawDisplay.SelectNormalArray = 'None'
domainrawDisplay.SelectTangentArray = 'None'
domainrawDisplay.SelectTCoordArray = 'None'
domainrawDisplay.TextureTransform = 'Transform2'
domainrawDisplay.OSPRayScaleArray = 'ImageFile'
domainrawDisplay.OSPRayScaleFunction = 'Piecewise Function'
domainrawDisplay.Assembly = ''
domainrawDisplay.SelectedBlockSelectors = ['']
domainrawDisplay.SelectOrientationVectors = 'None'
domainrawDisplay.ScaleFactor = 11.9
domainrawDisplay.SelectScaleArray = 'ImageFile'
domainrawDisplay.GlyphType = 'Arrow'
domainrawDisplay.GlyphTableIndexArray = 'ImageFile'
domainrawDisplay.GaussianRadius = 0.595
domainrawDisplay.SetScaleArray = ['POINTS', 'ImageFile']
domainrawDisplay.ScaleTransferFunction = 'Piecewise Function'
domainrawDisplay.OpacityArray = ['POINTS', 'ImageFile']
domainrawDisplay.OpacityTransferFunction = 'Piecewise Function'
domainrawDisplay.DataAxesGrid = 'Grid Axes Representation'
domainrawDisplay.PolarAxes = 'Polar Axes Representation'
domainrawDisplay.ScalarOpacityUnitDistance = 1.7320508075688774
domainrawDisplay.ScalarOpacityFunction = imageFilePWF
domainrawDisplay.TransferFunction2D = imageFileTF2D
domainrawDisplay.OpacityArrayName = ['POINTS', 'ImageFile']
domainrawDisplay.ColorArray2Name = ['POINTS', 'ImageFile']
domainrawDisplay.IsosurfaceValues = [0.5]
domainrawDisplay.SliceFunction = 'Plane'
domainrawDisplay.Slice = 59
domainrawDisplay.SelectInputVectors = [None, '']
domainrawDisplay.WriteLog = ''

# init the 'Plane' selected for 'SliceFunction'
domainrawDisplay.SliceFunction.Origin = [59.5, 59.5, 59.5]

# show data from clip2
clip2Display = Show(clip2, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
clip2Display.Representation = 'Surface'
clip2Display.ColorArrayName = ['POINTS', 'ImageFile']
clip2Display.LookupTable = imageFileLUT
clip2Display.SelectNormalArray = 'None'
clip2Display.SelectTangentArray = 'None'
clip2Display.SelectTCoordArray = 'None'
clip2Display.TextureTransform = 'Transform2'
clip2Display.OSPRayScaleArray = 'ImageFile'
clip2Display.OSPRayScaleFunction = 'Piecewise Function'
clip2Display.Assembly = ''
clip2Display.SelectedBlockSelectors = ['']
clip2Display.SelectOrientationVectors = 'None'
clip2Display.ScaleFactor = 12.700000000000001
clip2Display.SelectScaleArray = 'ImageFile'
clip2Display.GlyphType = 'Arrow'
clip2Display.GlyphTableIndexArray = 'ImageFile'
clip2Display.GaussianRadius = 0.635
clip2Display.SetScaleArray = ['POINTS', 'ImageFile']
clip2Display.ScaleTransferFunction = 'Piecewise Function'
clip2Display.OpacityArray = ['POINTS', 'ImageFile']
clip2Display.OpacityTransferFunction = 'Piecewise Function'
clip2Display.DataAxesGrid = 'Grid Axes Representation'
clip2Display.PolarAxes = 'Polar Axes Representation'
clip2Display.ScalarOpacityFunction = imageFilePWF
clip2Display.ScalarOpacityUnitDistance = 3.5578145209991225
clip2Display.OpacityArrayName = ['POINTS', 'ImageFile']
clip2Display.SelectInputVectors = [None, '']
clip2Display.WriteLog = ''

# show data from threshold1
threshold1Display = Show(threshold1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
threshold1Display.Representation = 'Surface'
threshold1Display.AmbientColor = [1.0, 0.6666666666666666, 1.0]
threshold1Display.ColorArrayName = ['POINTS', '']
threshold1Display.DiffuseColor = [1.0, 0.6666666666666666, 1.0]
threshold1Display.SelectNormalArray = 'None'
threshold1Display.SelectTangentArray = 'None'
threshold1Display.SelectTCoordArray = 'None'
threshold1Display.TextureTransform = 'Transform2'
threshold1Display.OSPRayScaleArray = 'ImageFile'
threshold1Display.OSPRayScaleFunction = 'Piecewise Function'
threshold1Display.Assembly = ''
threshold1Display.SelectedBlockSelectors = ['']
threshold1Display.SelectOrientationVectors = 'None'
threshold1Display.ScaleFactor = 12.700000000000001
threshold1Display.SelectScaleArray = 'ImageFile'
threshold1Display.GlyphType = 'Arrow'
threshold1Display.GlyphTableIndexArray = 'ImageFile'
threshold1Display.GaussianRadius = 0.635
threshold1Display.SetScaleArray = ['POINTS', 'ImageFile']
threshold1Display.ScaleTransferFunction = 'Piecewise Function'
threshold1Display.OpacityArray = ['POINTS', 'ImageFile']
threshold1Display.OpacityTransferFunction = 'Piecewise Function'
threshold1Display.DataAxesGrid = 'Grid Axes Representation'
threshold1Display.PolarAxes = 'Polar Axes Representation'
threshold1Display.ScalarOpacityUnitDistance = 4.05856534567142
threshold1Display.OpacityArrayName = ['POINTS', 'ImageFile']
threshold1Display.SelectInputVectors = [None, '']
threshold1Display.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
threshold1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
threshold1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# show data from clip1
clip1Display = Show(clip1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
clip1Display.Representation = 'Surface'
clip1Display.ColorArrayName = ['POINTS', 'ImageFile']
clip1Display.LookupTable = imageFileLUT
clip1Display.SelectNormalArray = 'None'
clip1Display.SelectTangentArray = 'None'
clip1Display.SelectTCoordArray = 'None'
clip1Display.TextureTransform = 'Transform2'
clip1Display.OSPRayScaleFunction = 'Piecewise Function'
clip1Display.Assembly = ''
clip1Display.SelectedBlockSelectors = ['']
clip1Display.SelectOrientationVectors = 'None'
clip1Display.ScaleFactor = -0.2
clip1Display.SelectScaleArray = 'None'
clip1Display.GlyphType = 'Arrow'
clip1Display.GlyphTableIndexArray = 'None'
clip1Display.GaussianRadius = -0.01
clip1Display.SetScaleArray = [None, '']
clip1Display.ScaleTransferFunction = 'Piecewise Function'
clip1Display.OpacityArray = [None, '']
clip1Display.OpacityTransferFunction = 'Piecewise Function'
clip1Display.DataAxesGrid = 'Grid Axes Representation'
clip1Display.PolarAxes = 'Polar Axes Representation'
clip1Display.ScalarOpacityFunction = imageFilePWF
clip1Display.OpacityArrayName = [None, '']
clip1Display.SelectInputVectors = [None, '']
clip1Display.WriteLog = ''

# show data from threshold2
threshold2Display = Show(threshold2, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
threshold2Display.Representation = 'Surface'
threshold2Display.ColorArrayName = ['POINTS', 'ImageFile']
threshold2Display.LookupTable = imageFileLUT
threshold2Display.SelectNormalArray = 'None'
threshold2Display.SelectTangentArray = 'None'
threshold2Display.SelectTCoordArray = 'None'
threshold2Display.TextureTransform = 'Transform2'
threshold2Display.OSPRayScaleArray = 'ImageFile'
threshold2Display.OSPRayScaleFunction = 'Piecewise Function'
threshold2Display.Assembly = ''
threshold2Display.SelectedBlockSelectors = ['']
threshold2Display.SelectOrientationVectors = 'None'
threshold2Display.ScaleFactor = 11.9
threshold2Display.SelectScaleArray = 'ImageFile'
threshold2Display.GlyphType = 'Arrow'
threshold2Display.GlyphTableIndexArray = 'ImageFile'
threshold2Display.GaussianRadius = 0.595
threshold2Display.SetScaleArray = ['POINTS', 'ImageFile']
threshold2Display.ScaleTransferFunction = 'Piecewise Function'
threshold2Display.OpacityArray = ['POINTS', 'ImageFile']
threshold2Display.OpacityTransferFunction = 'Piecewise Function'
threshold2Display.DataAxesGrid = 'Grid Axes Representation'
threshold2Display.PolarAxes = 'Polar Axes Representation'
threshold2Display.ScalarOpacityFunction = imageFilePWF
threshold2Display.ScalarOpacityUnitDistance = 4.065402085512077
threshold2Display.OpacityArrayName = ['POINTS', 'ImageFile']
threshold2Display.SelectInputVectors = [None, '']
threshold2Display.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
threshold2Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
threshold2Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# show data from clip3
clip3Display = Show(clip3, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
clip3Display.Representation = 'Surface'
clip3Display.AmbientColor = [1.0, 0.6666666666666666, 1.0]
clip3Display.ColorArrayName = ['POINTS', '']
clip3Display.DiffuseColor = [1.0, 0.6666666666666666, 1.0]
clip3Display.SelectNormalArray = 'None'
clip3Display.SelectTangentArray = 'None'
clip3Display.SelectTCoordArray = 'None'
clip3Display.TextureTransform = 'Transform2'
clip3Display.OSPRayScaleArray = 'ImageFile'
clip3Display.OSPRayScaleFunction = 'Piecewise Function'
clip3Display.Assembly = ''
clip3Display.SelectedBlockSelectors = ['']
clip3Display.SelectOrientationVectors = 'None'
clip3Display.ScaleFactor = 12.700000000000001
clip3Display.SelectScaleArray = 'ImageFile'
clip3Display.GlyphType = 'Arrow'
clip3Display.GlyphTableIndexArray = 'ImageFile'
clip3Display.GaussianRadius = 0.635
clip3Display.SetScaleArray = ['POINTS', 'ImageFile']
clip3Display.ScaleTransferFunction = 'Piecewise Function'
clip3Display.OpacityArray = ['POINTS', 'ImageFile']
clip3Display.OpacityTransferFunction = 'Piecewise Function'
clip3Display.DataAxesGrid = 'Grid Axes Representation'
clip3Display.PolarAxes = 'Polar Axes Representation'
clip3Display.ScalarOpacityUnitDistance = 3.5578145209991225
clip3Display.OpacityArrayName = ['POINTS', 'ImageFile']
clip3Display.SelectInputVectors = [None, '']
clip3Display.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
clip3Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
clip3Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# show data from threshold3
threshold3Display = Show(threshold3, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
threshold3Display.Representation = 'Surface'
threshold3Display.AmbientColor = [0.6, 0.5803921568627451, 0.6705882352941176]
threshold3Display.ColorArrayName = ['POINTS', '']
threshold3Display.DiffuseColor = [0.6, 0.5803921568627451, 0.6705882352941176]
threshold3Display.SelectNormalArray = 'None'
threshold3Display.SelectTangentArray = 'None'
threshold3Display.SelectTCoordArray = 'None'
threshold3Display.TextureTransform = 'Transform2'
threshold3Display.OSPRayScaleArray = 'ImageFile'
threshold3Display.OSPRayScaleFunction = 'Piecewise Function'
threshold3Display.Assembly = ''
threshold3Display.SelectedBlockSelectors = ['']
threshold3Display.SelectOrientationVectors = 'None'
threshold3Display.ScaleFactor = 11.9
threshold3Display.SelectScaleArray = 'ImageFile'
threshold3Display.GlyphType = 'Arrow'
threshold3Display.GlyphTableIndexArray = 'ImageFile'
threshold3Display.GaussianRadius = 0.595
threshold3Display.SetScaleArray = ['POINTS', 'ImageFile']
threshold3Display.ScaleTransferFunction = 'Piecewise Function'
threshold3Display.OpacityArray = ['POINTS', 'ImageFile']
threshold3Display.OpacityTransferFunction = 'Piecewise Function'
threshold3Display.DataAxesGrid = 'Grid Axes Representation'
threshold3Display.PolarAxes = 'Polar Axes Representation'
threshold3Display.ScalarOpacityUnitDistance = 1.9839551064698144
threshold3Display.OpacityArrayName = ['POINTS', 'ImageFile']
threshold3Display.SelectInputVectors = [None, '']
threshold3Display.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
threshold3Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
threshold3Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# show data from clip4
clip4Display = Show(clip4, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
clip4Display.Representation = 'Surface'
clip4Display.AmbientColor = [0.0, 0.6666666666666666, 1.0]
clip4Display.ColorArrayName = ['POINTS', '']
clip4Display.DiffuseColor = [0.0, 0.6666666666666666, 1.0]
clip4Display.SelectNormalArray = 'None'
clip4Display.SelectTangentArray = 'None'
clip4Display.SelectTCoordArray = 'None'
clip4Display.TextureTransform = 'Transform2'
clip4Display.OSPRayScaleArray = 'ImageFile'
clip4Display.OSPRayScaleFunction = 'Piecewise Function'
clip4Display.Assembly = ''
clip4Display.SelectedBlockSelectors = ['']
clip4Display.SelectOrientationVectors = 'None'
clip4Display.ScaleFactor = 12.700000000000001
clip4Display.SelectScaleArray = 'ImageFile'
clip4Display.GlyphType = 'Arrow'
clip4Display.GlyphTableIndexArray = 'ImageFile'
clip4Display.GaussianRadius = 0.635
clip4Display.SetScaleArray = ['POINTS', 'ImageFile']
clip4Display.ScaleTransferFunction = 'Piecewise Function'
clip4Display.OpacityArray = ['POINTS', 'ImageFile']
clip4Display.OpacityTransferFunction = 'Piecewise Function'
clip4Display.DataAxesGrid = 'Grid Axes Representation'
clip4Display.PolarAxes = 'Polar Axes Representation'
clip4Display.ScalarOpacityUnitDistance = 3.523392902114703
clip4Display.OpacityArrayName = ['POINTS', 'ImageFile']
clip4Display.SelectInputVectors = [None, '']
clip4Display.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
clip4Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
clip4Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for imageFileLUT in view renderView1
imageFileLUTColorBar = GetScalarBar(imageFileLUT, renderView1)
imageFileLUTColorBar.Title = 'ImageFile'
imageFileLUTColorBar.ComponentTitle = ''

# set color bar visibility
imageFileLUTColorBar.Visibility = 0

# hide data in view
Hide(geometry_paddedraw, renderView1)

# hide data in view
Hide(domainraw, renderView1)

# hide data in view
Hide(clip2, renderView1)

# hide data in view
Hide(clip1, renderView1)

# hide data in view
Hide(threshold2, renderView1)

# hide data in view
Hide(clip3, renderView1)

# ----------------------------------------------------------------
# setup color maps and opacity maps used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup animation scene, tracks and keyframes
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# initialize the timekeeper

# get time animation track
timeAnimationCue1 = GetTimeTrack()

# initialize the animation track

# get animation scene
animationScene1 = GetAnimationScene()

# initialize the animation scene
animationScene1.ViewModules = renderView1
animationScene1.Cues = timeAnimationCue1
animationScene1.AnimationTime = 0.0

# initialize the animation scene

# ----------------------------------------------------------------
# restore active source
SetActiveSource(None)
# ----------------------------------------------------------------


##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, usefull when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple splitted view)
# SaveScreenshot("path/to/screenshot.png", GetLayout())
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://www.paraview.org/paraview-docs/latest/python/paraview.simple.html
##--------------------------------------------