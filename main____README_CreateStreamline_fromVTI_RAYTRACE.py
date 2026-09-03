import os
import glob
import subprocess
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from paraview.simple import *
import numpy as np
from vtk.numpy_interface import dataset_adapter as dsa

# ==============================================================================
# ADDED: CUDA CHECK FOR OPTIX[cite: 2]
# ==============================================================================
def cuda_available():
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        subprocess.check_output(
            ["nvidia-smi", "-L"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return True
    except Exception:
        return False

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

def create_combined_subplot(img1, img2, img3, output_path):
    """Stitches 3 images side-by-side using matplotlib"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    titles = ["Solid + Streamlines", "Volume + Streamlines", "Solid Only"]
    image_paths = [img1, img2, img3]
    
    for ax, img_path, title in zip(axes, image_paths, titles):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=18, fontname='serif', pad=10)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" -> Saved Combined Subplot: {output_path}")

def plot_and_save_views(vti_filepath, dir_name, folder_name, solid_color=[0.5, 0.5, 0.5], show_colorbar=True):
    ResetSession()

    reader = XMLImageDataReader(FileName=[vti_filepath])
    reader.PointArrayStatus = ['Density', 'Velocity']
    reader.UpdatePipeline()

    view = CreateRenderView()
    view.ViewSize = [1200, 1200]
    view.Background = [1.0, 1.0, 1.0] 
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    
    # ==============================================================================
    # ADDED: ADVANCED RAY TRACING SETTINGS[cite: 2]
    # ==============================================================================
    view.EnableRayTracing = 1
    view.SamplesPerPixel = 40
    view.AmbientSamples = 5
    
    backend = "OSPRay pathtracer"
    try:
        available = list(view.GetProperty("BackEnd").GetAvailable())
        if "OptiX pathtracer" in available and cuda_available():
            backend = "OptiX pathtracer"
    except Exception:
        pass
    
    view.BackEnd = backend
    print(f" -> Using ray-tracing backend: {backend}")
    
    view.Shadows = 1
    if hasattr(view, "UseToneMapping"):
        view.UseToneMapping = 0
    if hasattr(view, "EnableOSPRayDenoiser"):
        view.EnableOSPRayDenoiser = 1

    # ==========================================
    # PIPELINE 1: SOLID CLIPS
    # ==========================================
    thresh = Threshold(Input=reader)
    thresh.Scalars = ['POINTS', 'Density']
    thresh.ThresholdMethod = 'Between'
    thresh.LowerThreshold = -1e10  
    thresh.UpperThreshold = 0.0

    clip1 = Clip(Input=thresh); clip1.ClipType = 'Plane'; clip1.ClipType.Normal, clip1.ClipType.Origin = [-1.0, 0.0, 0.0], [59.5, 59.5, 59.5] 
    clip2 = Clip(Input=thresh); clip2.ClipType = 'Plane'; clip2.ClipType.Normal, clip2.ClipType.Origin = [0.0, 1.0, 0.0], [59.5, 59.5, 59.5]
    clip3 = Clip(Input=thresh); clip3.ClipType = 'Plane'; clip3.ClipType.Normal, clip3.ClipType.Origin = [0.0, 0.0, 1.0], [59.5, 59.5, 59.5]

    disp_clips = []
    for clip in [clip1, clip2, clip3]:
        disp = Show(clip, view)
        disp.ColorArrayName = ['POINTS', '']  
        disp.DiffuseColor = solid_color
        disp.AmbientColor = solid_color
        disp.Opacity = 1.0
        
        # Ray-Tracing properties for the solid surface[cite: 2]
        disp.Specular = 0.5
        disp.SpecularPower = 10
        disp_clips.append(disp)

    # ==========================================
    # PIPELINE 2: WIREFRAME
    # ==========================================
    wireframe_tubes = _build_custom_wireframe()
    disp_frames = []
    for t in wireframe_tubes:
        disp_frame = Show(t, view)
        disp_frame.ColorArrayName = ['POINTS', '']  
        disp_frame.DiffuseColor = [0.0, 0.0, 0.0]  
        disp_frame.AmbientColor = [0.0, 0.0, 0.0]
        disp_frames.append(disp_frame)

    # ==========================================
    # PIPELINE 3: STREAMLINES
    # ==========================================
    stream = StreamTracer(Input=reader, SeedType='Point Cloud')
    stream.Vectors = ['POINTS', 'Velocity']
    stream.MaximumStreamlineLength = 1000.0 
    stream.SeedType.Center = [59.5, 59.5, 59.5]
    stream.SeedType.Radius = 120.0
    stream.SeedType.NumberOfPoints = 4000

    disp_stream = Show(stream, view)
    ColorBy(disp_stream, ('POINTS', 'Velocity', 'Magnitude'))
    
    velocityLUT = GetColorTransferFunction('Velocity')
    velocityLUT.ApplyPreset('Plasma (matplotlib)', True) 

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

    # ==========================================
    # PIPELINE 4: VOLUME RENDERING
    # ==========================================
    disp_vol = Show(reader, view)
    disp_vol.Representation = 'Volume'
    ColorBy(disp_vol, ('POINTS', 'Velocity', 'Magnitude'))
    
    # Setup opacity to go from Transparent (Velocity=0) to Opaque (Max Velocity)
    vel_info = reader.PointData.GetArray('Velocity')
    max_vel = vel_info.GetRange(-1)[1] if vel_info else 1.0 # -1 gets the magnitude range
    
    velocityPWF = GetOpacityTransferFunction('Velocity')
    # Points format: [Value, Opacity, Midpoint, Sharpness]
    velocityPWF.Points = [0.0, 0.0, 0.5, 0.0,   
                          max_vel, 1.0, 0.5, 0.0]

    # Ray-Tracing material properties for the fluid volume[cite: 2]
    disp_vol.Specular = 0.5
    disp_vol.SpecularPower = 100
    disp_vol.OSPRayMaterial = 'Water'

    _config_camera(view)

    # ==========================================
    # RENDER & SAVE EXPORTS
    # ==========================================
    # Define paths
    img1_path = os.path.join(dir_name, f"{folder_name}_1_Solid_Stream.png")
    img2_path = os.path.join(dir_name, f"{folder_name}_2_Volume_Stream.png")
    img3_path = os.path.join(dir_name, f"{folder_name}_3_Solid_Only.png")
    img_combined = os.path.join(dir_name, f"{folder_name}_Combined.png")

    # Image 1: Solid + Streamtrace
    for d in disp_clips: d.Visibility = 1
    disp_stream.Visibility = 1
    disp_vol.Visibility = 0
    Render()
    SaveScreenshot(img1_path, view, ImageResolution=[1200, 1200], TransparentBackground=0)

    # Image 2: Volume + Streamtrace
    for d in disp_clips: d.Visibility = 0
    disp_stream.Visibility = 1
    disp_vol.Visibility = 1
    Render()
    SaveScreenshot(img2_path, view, ImageResolution=[1200, 1200], TransparentBackground=0)

    # Image 3: Solid Only
    for d in disp_clips: d.Visibility = 1
    disp_stream.Visibility = 0
    disp_vol.Visibility = 0
    disp_stream.SetScalarBarVisibility(view, False) # Hide colorbar for solid only
    Render()
    SaveScreenshot(img3_path, view, ImageResolution=[1200, 1200], TransparentBackground=0)

    # Combine into Subplot
    create_combined_subplot(img1_path, img2_path, img3_path, img_combined)


# ==========================================
# MAIN EXECUTION
# ==========================================
BASE_DIR = "./Example_Bentheimer/"
FILE     = "output_data.vti"

# Recursively find all generated output_data.vti files
output_files = []
for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file == FILE:
            output_files.append(os.path.join(root, file))
            
if not output_files:
    print("No output_data.vti files found. Run the prediction script.")
    exit()
    
print(f"Found {len(output_files)} VTI files. Beginning batch render...")

for vti_path in output_files:
    dir_name = os.path.dirname(vti_path)
    folder_name = os.path.basename(dir_name)
    
    print(f" -> Processing folder: {folder_name}")
    plot_and_save_views(
        vti_filepath=vti_path, 
        dir_name=dir_name,
        folder_name=folder_name,
        show_colorbar=True
    )
    
print("\nBatch rendering complete!")