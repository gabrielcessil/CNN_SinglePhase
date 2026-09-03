import os
import shutil
import subprocess
import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from paraview.simple import *


# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = "./Example_Bentheimer/"
FILE = "output_data.vti"

OUTPUT_DIR = "./rotation_video/"
FRAME_DIR = os.path.join(OUTPUT_DIR, "frames")
VIDEO_FILE = os.path.join(OUTPUT_DIR, "rotating_views.mp4")

# Video
N_FRAMES = 120
FPS = 30
ROTATION_DEG = 360.0

# Render resolution of EACH individual view
VIEW_RESOLUTION = [1200, 1200]

# Final side-by-side resolution is approximately 3600 x 1200
DPI = 120

# Fixed camera: this is NEVER changed during the animation.
CAMERA_POSITION = [-145.62500334229827, 258.12891252294224, 337.27420872575703]
CAMERA_FOCAL_POINT = [54.23265623503145, 49.59503372400067, 63.19529352193284]
CAMERA_VIEW_UP = [0.28077913995658044, 0.8513959344245512, -0.4430440580919559]
CAMERA_VIEW_ANGLE = 30
CAMERA_PARALLEL_SCALE = 103.05702305034819

# Object rotation center
ROTATION_CENTER = np.array([59.5, 59.5, 59.5], dtype=float)

SOLID_COLOR = [0.5, 0.5, 0.5]

SHOW_COLORBAR = True


# ==============================================================================
# OPTIONAL CUDA / OPTIX CHECK
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


# ==============================================================================
# CAMERA
# ==============================================================================

def configure_static_camera(view):
    """
    The camera remains completely static for the whole animation.

    The rotation is applied to the rendered objects, NOT to the camera.
    """
    view.CameraPosition = CAMERA_POSITION
    view.CameraFocalPoint = CAMERA_FOCAL_POINT
    view.CameraViewUp = CAMERA_VIEW_UP
    view.CameraViewAngle = CAMERA_VIEW_ANGLE
    view.CameraParallelScale = CAMERA_PARALLEL_SCALE
    view.CenterOfRotation = ROTATION_CENTER.tolist()


# ==============================================================================
# WIREFRAME
# ==============================================================================

def build_custom_wireframe():
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
        ([0.0, 0.0, 119.0], [0.0, 59.5, 119.0]),
    ]

    tubes = []

    for pt1, pt2 in wireframe_coords:
        line = Line(Point1=pt1, Point2=pt2)
        tube = Tube(Input=line)
        tube.Radius = 0.3
        tube.Capping = 1
        tubes.append(tube)

    return tubes


# ==============================================================================
# ROTATION
# ==============================================================================

def apply_rotation(transform_filter, angle_deg):
    """
    Rotate an object around ROTATION_CENTER.

    This is the important difference from make_images_cameramove7.py:
    there the camera position changes; here the camera stays fixed and
    the rendered object rotates.
    """
    transform_filter.Transform.Rotate = [0.0, 0.0, angle_deg]
    transform_filter.Transform.Translate = [
        0.0,
        0.0,
        0.0,
    ]

    # ParaView's Transform filter rotates around its origin.
    # Move the object so the rotation happens around ROTATION_CENTER.
    transform_filter.Transform.Translate = (
        ROTATION_CENTER -
        np.array(transform_filter.GetDataInformation().GetBounds()[::2])
    ).tolist()


def make_rotation_transform(source, angle_deg):
    """
    Create a Transform filter that rotates source around ROTATION_CENTER.

    ParaView Transform uses the object's origin. We therefore explicitly
    construct the equivalent rotation around the desired center:
        x' = C + R(x - C)
    using Translate(C), Rotate(angle), Translate(-C).
    """

    transform = Transform(Input=source)

    # The Transform filter applies operations in its transform.
    # Use the standard ParaView center-of-rotation fields when available.
    try:
        transform.Transform.Center = ROTATION_CENTER.tolist()
    except Exception:
        pass

    transform.Transform.Rotate = [0.0, 0.0, angle_deg]

    return transform


# ==============================================================================
# PIPELINE CREATION
# ==============================================================================

def build_scene(vti_filepath, view):
    reader = XMLImageDataReader(FileName=[vti_filepath])
    reader.PointArrayStatus = ["Density", "Velocity"]
    reader.UpdatePipeline()

    objects = [reader]

    # --------------------------------------------------------------------------
    # SOLID
    # --------------------------------------------------------------------------
    thresh = Threshold(Input=reader)
    thresh.Scalars = ["POINTS", "Density"]
    thresh.ThresholdMethod = "Between"
    thresh.LowerThreshold = -1e10
    thresh.UpperThreshold = 0.0
    objects.append(thresh)

    clips = []

    clip1 = Clip(Input=thresh)
    clip1.ClipType = "Plane"
    clip1.ClipType.Normal = [-1.0, 0.0, 0.0]
    clip1.ClipType.Origin = ROTATION_CENTER.tolist()

    clip2 = Clip(Input=thresh)
    clip2.ClipType = "Plane"
    clip2.ClipType.Normal = [0.0, 1.0, 0.0]
    clip2.ClipType.Origin = ROTATION_CENTER.tolist()

    clip3 = Clip(Input=thresh)
    clip3.ClipType = "Plane"
    clip3.ClipType.Normal = [0.0, 0.0, 1.0]
    clip3.ClipType.Origin = ROTATION_CENTER.tolist()

    clips.extend([clip1, clip2, clip3])
    objects.extend(clips)

    solid_displays = []

    for clip in clips:
        disp = Show(clip, view)
        disp.ColorArrayName = ["POINTS", ""]
        disp.DiffuseColor = SOLID_COLOR
        disp.AmbientColor = SOLID_COLOR
        disp.Opacity = 1.0
        disp.Specular = 0.5
        disp.SpecularPower = 10
        solid_displays.append(disp)

    # --------------------------------------------------------------------------
    # WIREFRAME
    # --------------------------------------------------------------------------
    wireframe_sources = build_custom_wireframe()
    objects.extend(wireframe_sources)

    wireframe_displays = []

    for source in wireframe_sources:
        disp = Show(source, view)
        disp.ColorArrayName = ["POINTS", ""]
        disp.DiffuseColor = [0.0, 0.0, 0.0]
        disp.AmbientColor = [0.0, 0.0, 0.0]
        wireframe_displays.append(disp)

    # --------------------------------------------------------------------------
    # STREAMLINES
    # --------------------------------------------------------------------------
    stream = StreamTracer(Input=reader, SeedType="Point Cloud")
    stream.Vectors = ["POINTS", "Velocity"]
    stream.MaximumStreamlineLength = 1000.0
    stream.SeedType.Center = ROTATION_CENTER.tolist()
    stream.SeedType.Radius = 120.0
    stream.SeedType.NumberOfPoints = 4000
    objects.append(stream)

    disp_stream = Show(stream, view)
    ColorBy(disp_stream, ("POINTS", "Velocity", "Magnitude"))

    velocity_lut = GetColorTransferFunction("Velocity")
    velocity_lut.ApplyPreset("Plasma (matplotlib)", True)

    if SHOW_COLORBAR:
        disp_stream.SetScalarBarVisibility(view, True)
        color_bar = GetScalarBar(velocity_lut, view)
        color_bar.TitleColor = [0.0, 0.0, 0.0]
        color_bar.LabelColor = [0.0, 0.0, 0.0]
        color_bar.TitleFontFamily = "Times"
        color_bar.LabelFontFamily = "Times"
        color_bar.TitleFontSize = 28
        color_bar.LabelFontSize = 28
        color_bar.AutomaticLabelFormat = 0
        color_bar.LabelFormat = "%.2e"
    else:
        disp_stream.SetScalarBarVisibility(view, False)

    # --------------------------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------------------------
    disp_vol = Show(reader, view)
    disp_vol.Representation = "Volume"
    ColorBy(disp_vol, ("POINTS", "Velocity", "Magnitude"))

    vel_info = reader.PointData.GetArray("Velocity")
    max_vel = vel_info.GetRange(-1)[1] if vel_info else 1.0

    velocity_pwf = GetOpacityTransferFunction("Velocity")
    velocity_pwf.Points = [
        0.0, 0.0, 0.5, 0.0,
        max_vel, 1.0, 0.5, 0.0,
    ]

    disp_vol.Specular = 0.5
    disp_vol.SpecularPower = 100
    disp_vol.OSPRayMaterial = "Water"

    return {
        "reader": reader,
        "objects": objects,
        "solid": solid_displays,
        "wireframe": wireframe_displays,
        "stream": disp_stream,
        "volume": disp_vol,
        "lut": velocity_lut,
        "pwf": velocity_pwf,
    }


# ==============================================================================
# TRANSFORM ALL VISUAL OBJECTS
# ==============================================================================

def create_rotated_scene(scene, angle_deg, view):
    """
    Creates transformed copies of the visible objects.

    The source data remains unchanged. Only the displayed geometry/volume
    is rotated around ROTATION_CENTER.
    """

    transformed = []

    # Solid clips
    for disp in scene["solid"]:
        source = disp.Input
        tr = Transform(Input=source)

        try:
            tr.Transform.Center = ROTATION_CENTER.tolist()
        except Exception:
            pass

        tr.Transform.Rotate = [0.0, 0.0, angle_deg]

        new_disp = Show(tr, view)
        new_disp.ColorArrayName = ["POINTS", ""]
        new_disp.DiffuseColor = SOLID_COLOR
        new_disp.AmbientColor = SOLID_COLOR
        new_disp.Opacity = 1.0
        new_disp.Specular = 0.5
        new_disp.SpecularPower = 10

        transformed.append((tr, new_disp, "solid"))

    # Wireframe
    for disp in scene["wireframe"]:
        source = disp.Input
        tr = Transform(Input=source)

        try:
            tr.Transform.Center = ROTATION_CENTER.tolist()
        except Exception:
            pass

        tr.Transform.Rotate = [0.0, 0.0, angle_deg]

        new_disp = Show(tr, view)
        new_disp.ColorArrayName = ["POINTS", ""]
        new_disp.DiffuseColor = [0.0, 0.0, 0.0]
        new_disp.AmbientColor = [0.0, 0.0, 0.0]

        transformed.append((tr, new_disp, "wireframe"))

    # Streamlines
    stream_source = scene["stream"].Input
    tr_stream = Transform(Input=stream_source)

    try:
        tr_stream.Transform.Center = ROTATION_CENTER.tolist()
    except Exception:
        pass

    tr_stream.Transform.Rotate = [0.0, 0.0, angle_deg]

    disp_stream = Show(tr_stream, view)
    ColorBy(disp_stream, ("POINTS", "Velocity", "Magnitude"))
    disp_stream.SetScalarBarVisibility(view, SHOW_COLORBAR)

    transformed.append((tr_stream, disp_stream, "stream"))

    # Volume
    volume_source = scene["reader"]
    tr_vol = Transform(Input=volume_source)

    try:
        tr_vol.Transform.Center = ROTATION_CENTER.tolist()
    except Exception:
        pass

    tr_vol.Transform.Rotate = [0.0, 0.0, angle_deg]

    disp_vol = Show(tr_vol, view)
    disp_vol.Representation = "Volume"
    ColorBy(disp_vol, ("POINTS", "Velocity", "Magnitude"))
    disp_vol.Specular = 0.5
    disp_vol.SpecularPower = 100
    disp_vol.OSPRayMaterial = "Water"

    transformed.append((tr_vol, disp_vol, "volume"))

    return transformed


# ==============================================================================
# RENDER ONE OF THE THREE PANELS
# ==============================================================================

def render_panel(view, scene, transformed, panel_type, output_path):
    # Hide every transformed display first
    for _, disp, _ in transformed:
        disp.Visibility = 0

    if panel_type == "solid_stream":
        for _, disp, kind in transformed:
            if kind in ("solid", "wireframe", "stream"):
                disp.Visibility = 1

    elif panel_type == "volume_stream":
        for _, disp, kind in transformed:
            if kind in ("stream", "volume"):
                disp.Visibility = 1

    elif panel_type == "solid_only":
        for _, disp, kind in transformed:
            if kind in ("solid", "wireframe"):
                disp.Visibility = 1

        for _, disp, kind in transformed:
            if kind == "stream":
                disp.SetScalarBarVisibility(view, False)

    Render()

    SaveScreenshot(
        output_path,
        view,
        ImageResolution=VIEW_RESOLUTION,
        TransparentBackground=0,
        OverrideColorPalette="WhiteBackground",
    )


# ==============================================================================
# COMBINE THE THREE PANELS
# ==============================================================================

def create_combined_image(img1, img2, img3, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    titles = [
        "Solid + Streamlines",
        "Volume + Streamlines",
        "Solid Only",
    ]

    for ax, image_path, title in zip(
        axes,
        [img1, img2, img3],
        titles,
    ):
        img = mpimg.imread(image_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=18, fontname="serif", pad=10)
        ax.axis("off")

    plt.tight_layout(pad=0.5)
    plt.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)


# ==============================================================================
# VIDEO CREATION
# ==============================================================================

def make_video(frame_pattern, output_video):
    """
    Uses FFmpeg to convert the rendered PNG sequence into MP4.
    """

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found in PATH. Install/load ffmpeg before running."
        )

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(FPS),
        "-i", frame_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "medium",
        "-movflags", "+faststart",
        output_video,
    ]

    print("\nCreating video...")
    subprocess.run(cmd, check=True)
    print(f"Video saved: {output_video}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FRAME_DIR, exist_ok=True)

    vti_path = os.path.join(BASE_DIR, FILE)

    if not os.path.exists(vti_path):
        raise FileNotFoundError(
            f"Could not find VTI file:\n{vti_path}"
        )

    print("=" * 70)
    print("STATIC CAMERA / 360 DEGREE OBJECT ROTATION")
    print("=" * 70)
    print(f"Input:       {vti_path}")
    print(f"Frames:      {N_FRAMES}")
    print(f"FPS:         {FPS}")
    print(f"Duration:    {N_FRAMES / FPS:.2f} s")
    print(f"Rotation:    {ROTATION_DEG} degrees")
    print("Camera:      STATIC")
    print(f"Output:      {VIDEO_FILE}")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # Render view
    # --------------------------------------------------------------------------
    ResetSession()

    view = CreateRenderView()
    view.ViewSize = VIEW_RESOLUTION
    view.Background = [1.0, 1.0, 1.0]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0

    # Same rendering philosophy as the original script
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

    try:
        view.BackEnd = backend
    except Exception:
        pass

    view.Shadows = 1

    if hasattr(view, "UseToneMapping"):
        view.UseToneMapping = 0

    if hasattr(view, "EnableOSPRayDenoiser"):
        view.EnableOSPRayDenoiser = 1

    print(f"Ray-tracing backend: {backend}")

    # IMPORTANT: camera is configured only once.
    configure_static_camera(view)

    # Build original scene once
    scene = build_scene(vti_path, view)

    # Hide original objects because we will display rotated transforms.
    for disp in scene["solid"]:
        disp.Visibility = 0

    for disp in scene["wireframe"]:
        disp.Visibility = 0

    scene["stream"].Visibility = 0
    scene["volume"].Visibility = 0

    # --------------------------------------------------------------------------
    # Generate rotation frames
    # --------------------------------------------------------------------------
    angles = np.linspace(
        0.0,
        ROTATION_DEG,
        N_FRAMES,
        endpoint=False,
    )

    for frame_idx, angle in enumerate(angles):

        print(
            f"[{frame_idx + 1:03d}/{N_FRAMES}] "
            f"rotation = {angle:7.2f} deg",
            flush=True,
        )

        # Create transformed visual objects for this frame
        transformed = create_rotated_scene(
            scene,
            float(angle),
            view,
        )

        frame_number = f"{frame_idx:04d}"

        img1 = os.path.join(
            FRAME_DIR,
            f"frame_{frame_number}_solid_stream.png",
        )

        img2 = os.path.join(
            FRAME_DIR,
            f"frame_{frame_number}_volume_stream.png",
        )

        img3 = os.path.join(
            FRAME_DIR,
            f"frame_{frame_number}_solid_only.png",
        )

        combined = os.path.join(
            FRAME_DIR,
            f"frame_{frame_number}.png",
        )

        # Render all three side-by-side images
        render_panel(
            view,
            scene,
            transformed,
            "solid_stream",
            img1,
        )

        render_panel(
            view,
            scene,
            transformed,
            "volume_stream",
            img2,
        )

        render_panel(
            view,
            scene,
            transformed,
            "solid_only",
            img3,
        )

        create_combined_image(
            img1,
            img2,
            img3,
            combined,
        )

        # Delete transformed objects before next frame
        for tr, disp, _ in reversed(transformed):
            try:
                Delete(disp)
            except Exception:
                pass

            try:
                Delete(tr)
            except Exception:
                pass

        gc.collect()

    # --------------------------------------------------------------------------
    # Create MP4
    # --------------------------------------------------------------------------
    make_video(
        os.path.join(FRAME_DIR, "frame_%04d.png"),
        VIDEO_FILE,
    )

    # --------------------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------------------
    try:
        Delete(scene["reader"])
    except Exception:
        pass

    try:
        Delete(view)
    except Exception:
        pass

    Disconnect()
    gc.collect()

    print("\n" + "=" * 70)
    print("DONE")
    print(f"Video: {VIDEO_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
