import os
import glob
from paraview.simple import *

def _config_camera(view):
    view.CameraPosition = [-145.62500334229827, 258.12891252294224, 337.27420872575703]
    view.CameraFocalPoint = [54.23265623503145, 49.59503372400067, 63.19529352193284]
    view.CameraViewUp = [0.28077913995658044, 0.8513959344245512, -0.4430440580919559]
    view.CameraViewAngle = 30
    view.CameraParallelScale = 103.05702305034819
    view.CenterOfRotation = [59.5, 59.5, 59.5]

def _build_custom_wireframe():
    wireframe_coords = [
        ([59.5, 59.5, 59.5], [0.0, 59.5, 59.5]),
        ([59.5, 59.5, 59.5], [59.5, 119.0, 59.5]),
        ([59.5, 59.5, 59.5], [59.5, 59.5, 119.0]),
        ([0.0, 59.5, 59.5], [0.0, 119.0, 59.5]),
        ([0.0, 59.5, 59.5], [0.0, 59.5, 119.0]),
        ([59.5, 119.0, 59.5], [0.0, 119.0, 59.5]),
        ([59.5, 119.0, 59.5], [59.5, 119.0, 119.0]),
        ([59.5, 59.5, 119.0], [0.0, 59.5, 119.0]),
        ([59.5, 59.5, 119.0], [59.5, 119.0, 119.0]),
        ([0.0, 0.0, 0.0], [119.0, 0.0, 0.0]),
        ([0.0, 119.0, 0.0], [119.0, 119.0, 0.0]),
        ([0.0, 0.0, 119.0], [119.0, 0.0, 119.0]),
        ([0.0, 0.0, 0.0], [0.0, 119.0, 0.0]),
        ([119.0, 0.0, 0.0], [119.0, 119.0, 0.0]),
        ([119.0, 0.0, 119.0], [119.0, 119.0, 119.0]),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 119.0]),
        ([119.0, 0.0, 0.0], [119.0, 0.0, 119.0]),
        ([119.0, 119.0, 0.0], [119.0, 119.0, 119.0]),
        ([0.0, 119.0, 0.0], [0.0, 119.0, 59.5]),
        ([119.0, 119.0, 119.0], [59.5, 119.0, 119.0]),
        ([0.0, 0.0, 119.0], [0.0, 59.5, 119.0])
    ]
    tubes = []
    for pt1, pt2 in wireframe_coords:
        l = Line(Point1=pt1, Point2=pt2)
        t = Tube(Input=l)
        t.Radius = 0.3
        t.Capping = 1
        tubes.append(t)
    return tubes

def plot_streamlines(vti_filepath, output_image_path, solid_color=[0.5, 0.5, 0.5], show_colorbar=True):
    ResetSession()

    reader = XMLImageDataReader(FileName=[vti_filepath])
    reader.PointArrayStatus = ['Density', 'Velocity']
    reader.UpdatePipeline()

    view = CreateRenderView()
    view.ViewSize = [1200, 1200]
    view.Background = [1.0, 1.0, 1.0] 
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    view.EnableRayTracing = 1
    view.Shadows = 0                
    view.SamplesPerPixel = 40       
    view.AmbientSamples = 2         
    if hasattr(view, "EnableOSPRayDenoiser"):
        view.EnableOSPRayDenoiser = 1

    thresh = Threshold(Input=reader)
    thresh.Scalars = ['POINTS', 'Density']
    thresh.ThresholdMethod = 'Between'
    thresh.LowerThreshold = -1e10  
    thresh.UpperThreshold = 0.0

    clip1 = Clip(Input=thresh)
    clip1.ClipType = 'Plane'
    clip1.ClipType.Normal, clip1.ClipType.Origin = [-1.0, 0.0, 0.0], [59.5, 59.5, 59.5] 

    clip2 = Clip(Input=thresh)
    clip2.ClipType = 'Plane'
    clip2.ClipType.Normal, clip2.ClipType.Origin = [0.0, 1.0, 0.0], [59.5, 59.5, 59.5]
    
    clip3 = Clip(Input=thresh)
    clip3.ClipType = 'Plane'
    clip3.ClipType.Normal, clip3.ClipType.Origin = [0.0, 0.0, 1.0], [59.5, 59.5, 59.5]

    for clip in [clip1, clip2, clip3]:
        disp = Show(clip, view)
        disp.ColorArrayName = ['POINTS', '']  
        disp.DiffuseColor = solid_color
        disp.AmbientColor = solid_color
        disp.Opacity = 1.0

    wireframe_tubes = _build_custom_wireframe()
    for t in wireframe_tubes:
        disp_frame = Show(t, view)
        disp_frame.ColorArrayName = ['POINTS', '']  
        disp_frame.DiffuseColor = [0.0, 0.0, 0.0]  
        disp_frame.AmbientColor = [0.0, 0.0, 0.0]

    stream = StreamTracer(Input=reader, SeedType='Point Cloud')
    stream.Vectors = ['POINTS', 'Velocity']
    stream.MaximumStreamlineLength = 1000.0 
    stream.SeedType.Center = [59.5, 59.5, 59.5]
    stream.SeedType.Radius = 120.0
    stream.SeedType.NumberOfPoints = 1150

    disp_stream = Show(stream, view)
    ColorBy(disp_stream, ('POINTS', 'Velocity', 'Magnitude'))
    
    velocityLUT = GetColorTransferFunction('Velocity')
    velocityLUT.ApplyPreset('autumn (matplotlib)', True) 

    if show_colorbar:
        disp_stream.SetScalarBarVisibility(view, True)
        color_bar = GetScalarBar(velocityLUT, view)
        color_bar.TitleColor = [0.0, 0.0, 0.0]
        color_bar.LabelColor = [0.0, 0.0, 0.0]
        color_bar.TitleFontFamily = 'Times'
        color_bar.LabelFontFamily = 'Times'
        color_bar.TitleFontSize = 28
        color_bar.LabelFontSize = 28
        color_bar.AutomaticLabelFormat = 0
        color_bar.LabelFormat = '%.2e'
    else:
        disp_stream.SetScalarBarVisibility(view, False)

    _config_camera(view)
    
    Render() 
    SaveScreenshot(output_image_path, view, ImageResolution=[1200, 1200], TransparentBackground=0)

    
# 1. Define where to search for the VTI files
BASE_DIR = "./Example_Bentheimer/"
FILE     = "output_data.vti"

# 2. Recursively find all generated output_data.vti files
target_files = []
for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file == FILE:
            target_files.append(os.path.join(root, file))
            
if not target_files:
    print("No output_data.vti files found. Did you run the prediction script first?")
    exit()
    
print(f"Found {len(target_files)} VTI files. Beginning batch render...")

# 3. Render and save each file in its native directory
for vti_path in target_files:
    dir_name = os.path.dirname(vti_path)
    folder_name = os.path.basename(dir_name)
    
    out_image = os.path.join(dir_name, f"streamlines_{folder_name}.png")
    print(f" -> Rendering: {out_image}")
    
    plot_streamlines(
        vti_filepath=vti_path, 
        output_image_path=out_image, 
        show_colorbar=True
    )
    
print("\nBatch rendering complete!")