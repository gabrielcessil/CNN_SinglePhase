import os
import sys
import gc
import re
import json
import subprocess
import shutil
import concurrent.futures
import numpy as np

# We only import matplotlib inside the worker to prevent orchestrator overhead
if '--worker' in sys.argv:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

# ==============================================================================
# 1. CONDITIONAL PARAVIEW IMPORT
# ==============================================================================
if '--worker' in sys.argv or '--preview' in sys.argv:
    from paraview.simple import *

# ==============================================================================
# 2. IMMEDIATE-MODE RENDERER (Worker Mode)
# ==============================================================================
def cuda_available():
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        subprocess.check_output(
            ["nvidia-smi", "-L"], stderr=subprocess.DEVNULL, text=True, timeout=5
        )
        return True
    except Exception:
        return False

class ParaViewRenderer:
    def __init__(self, config):
        self.config = config
        self.view = None
        self.lut = None
        self.pwf = None

    def _setup_view(self):
        for v in GetRenderViews():
            try: Delete(v)
            except: pass
            gc.collect()
        
        self.view = CreateRenderView()
        self.view.ViewSize = self.config['resolution']
        
        print("\n=== VIEW CONFIGURATION PROPERTIES ===")
        for prop in self.view.ListProperties():
            print(f"{prop}: {self.view.GetPropertyValue(prop)}")
        print("=====================================\n")
        
        # Strict White Background from state file
        if hasattr(self.view, 'UseColorPaletteForBackground'):
            self.view.UseColorPaletteForBackground = 0
        self.view.Background = [1.0, 1.0, 1.0]
        self.view.Background2 = [1.0, 1.0, 1.0]
        self.view.OrientationAxesVisibility = 0
        
        # Raytracing Setup
        self.view.EnableRayTracing = 1
        self.view.SamplesPerPixel = self.config.get('samples', 4)
        self.view.AmbientSamples = self.config.get('ambient_samples', 6)
        
        backend = "OSPRay pathtracer"
        try:
            available = list(self.view.GetProperty("BackEnd").GetAvailable())
            if "OptiX pathtracer" in available and cuda_available():
                backend = "OptiX pathtracer"
        except Exception:
            pass
        print(f"Using {backend} as backend")
        self.view.BackEnd = backend
        
        if hasattr(self.view, 'Backgroundmode'):
            self.view.Backgroundmode = 'Backplate'
        else: print("OSPRayBackgroundMode not available")
        
        # Environment Lighting from state file (affects global illumination)
        if hasattr(self.view, 'EnvironmentalBG'):
            self.view.EnvironmentalBG = [0.6, 0.5803921568627451, 0.6705882352941176]
        else: print("EnvironmentalBG not available")
        
        # CRITICAL: Enable environment lighting so OSPRay uses ambient light correctly
        if hasattr(self.view, 'UseEnvironmentLighting'):
            self.view.UseEnvironmentLighting = 1
            
        # Orient the environmental lighting (these are the default vectors)
        if hasattr(self.view, 'EnvironmentNorth'):
            self.view.EnvironmentNorth = [0.0, 1.0, 0.0] 
        
        if hasattr(self.view, 'EnvironmentEast'):
            self.view.EnvironmentEast = [1.0, 0.0, 0.0]
            
        # Adjust the overall intensity of the environmental light
        if hasattr(self.view, 'LightScale'):
            self.view.LightScale = 1.0  
        
        if hasattr(self.view, 'RouletteDepth'):
            self.view.RouletteDepth = 5
        else: print("RouletteDepth not available")
            
        self.view.Shadows = 1
        
        if hasattr(self.view, 'ProgressivePasses'):
            self.view.ProgressivePasses = 3
        else: print("ProgressivePasses not available")
            
        # Disable tonemapping to match raw ParaView 5.13 state file colors
        if hasattr(self.view, "UseToneMapping"):
            self.view.UseToneMapping = 0
        else: print("UseToneMapping not available")
        
        if hasattr(self.view, 'Denoise'):
            self.view.Denoise = 1
        else: print("Denoise not available")
        
        if hasattr(self.view, 'UseLight'):
            self.view.UseLight = 1
        else: print("UseLight not available")
        
        if hasattr(self.view, 'ScalingMode'):
            self.view.ScalingMode = 'All Approximate'
        else: print("ScalingMode not available")
        
    def render_file(self, file_path, output_filename, frame_idx, override_camera_pos, override_view_up):
        self._setup_view()
        
        # 1. Load Data
        if file_path.endswith('.pvti'):
            reader = XMLPartitionedImageDataReader(FileName=[file_path])
        else:
            reader = XMLImageDataReader(FileName=[file_path])
            
        reader.PointArrayStatus = [self.config['scalar_name'], self.config['vector_name']]
        reader.UpdatePipeline()
        
        bounds = reader.GetDataInformation().GetBounds()
        center = [(bounds[0] + bounds[1])/2, (bounds[2] + bounds[3])/2, (bounds[4] + bounds[5])/2]
        
        objects_to_delete = [reader]
        
        # ----------------------------------------------------------------------
        # SOLID
        # ----------------------------------------------------------------------
        thresh = Threshold(Input=reader)
        thresh.Scalars = ['POINTS', self.config['scalar_name']]
        thresh.ThresholdMethod = 'Between'
        thresh.LowerThreshold = -1e10
        thresh.UpperThreshold = 0.0
        
        clip1 = Clip(Input=thresh)
        clip1.ClipType = "Plane"
        clip1.ClipType.Normal = [-1.0, 0.0, 0.0]
        clip1.ClipType.Origin = center

        clip2 = Clip(Input=thresh)
        clip2.ClipType = "Plane"
        clip2.ClipType.Normal = [0.0, 1.0, 0.0]
        clip2.ClipType.Origin = center

        clip3 = Clip(Input=thresh)
        clip3.ClipType = "Plane"
        clip3.ClipType.Normal = [0.0, 0.0, 1.0]
        clip3.ClipType.Origin = center
        
        objects_to_delete.extend([thresh, clip1, clip2, clip3])
        
        solid_displays = []
        for clip_obj in [clip1, clip2, clip3]:
            # DO NOT use ExtractSurface here! We need the 3D internal cells for Volume rendering.
            disp_solid = Show(clip_obj, self.view)
            
            # CRITICAL: Apply material properties IMMEDIATELY upon creation
            disp_solid.Representation = 'Surface'
            ColorBy(disp_solid, None) 
            disp_solid.ColorArrayName = [None, '']
            
            # Apply Gray-Green
            disp_solid.AmbientColor = [0.45, 0.55, 0.45]  # Slightly darker base
            disp_solid.DiffuseColor = [0.55, 0.65, 0.55]  # Main color under direct light
            disp_solid.Opacity = 1.0 
            
            # Keep Specular low for the matte finish
            disp_solid.Specular = 1.0          # Maximum reflection strength
            disp_solid.SpecularPower = 100.0   # Sharp, clear reflection
            disp_solid.OSPRayMaterial = "None"
            
            solid_displays.append(disp_solid)
            
        # Update the colormap to match the new gray-green      
        solid_lut = GetColorTransferFunction(self.config['scalar_name'])
        solid_lut.RGBPoints = [
            -1e10, 0.55, 0.62, 0.55,
             0.0,  0.55, 0.62, 0.55,
             1e10, 0.55, 0.62, 0.55
        ]
        solid_lut.ColorSpace = 'RGB'
        
        solid_pwf = GetOpacityTransferFunction(self.config['scalar_name'])
        
        # IMPORTANT:
        # Density <= 0  -> transparent
        # Density > 0   -> significant volumetric opacity
        solid_pwf.Points = [
            -1e10, 0.00, 0.5, 0.0,
             0.0,  0.00, 0.5, 0.0,
             1e10, 0.80, 0.5, 0.0
        ]
                    
        # ----------------------------------------------------------------------
        # WIREFRAME (Dynamic adaptive bounds)
        # ----------------------------------------------------------------------
        outline = Outline(Input=reader)
        tube = Tube(Input=outline)
        tube.Radius = 0.3
        tube.Capping = 1
        objects_to_delete.extend([outline, tube])
        
        disp_wire = Show(tube, self.view)
        disp_wire.ColorArrayName = ["POINTS", ""]
        disp_wire.DiffuseColor = [0.0, 0.0, 0.0]
        disp_wire.OSPRayMaterial = "None"
        
        # ----------------------------------------------------------------------
        # STREAMLINES
        # ----------------------------------------------------------------------
        stream = StreamTracer(Input=reader, SeedType="Point Cloud")
        stream.Vectors = ["POINTS", self.config['vector_name']]
        stream.MaximumStreamlineLength = 1000.0
        stream.SeedType.Center = center
        stream.SeedType.Radius = (bounds[1]-bounds[0]) # Auto-scale radius
        stream.SeedType.NumberOfPoints = 14000
        objects_to_delete.append(stream)

        disp_stream = Show(stream, self.view)
        ColorBy(disp_stream, ("POINTS", self.config['vector_name'], "Magnitude"))
        disp_stream.OSPRayMaterial = "None"
        
        self.lut = GetColorTransferFunction(self.config['vector_name'])
        self.lut.ApplyPreset("Plasma (matplotlib)", True)
        
        color_bar = GetScalarBar(self.lut, self.view)
        color_bar.TitleColor = [0.0, 0.0, 0.0]
        color_bar.LabelColor = [0.0, 0.0, 0.0]
        color_bar.TitleFontFamily = "Times"
        color_bar.LabelFontFamily = "Times"
        color_bar.TitleFontSize = 28
        color_bar.LabelFontSize = 28
        color_bar.AutomaticLabelFormat = 0
        color_bar.LabelFormat = "%.2e"
        
        # ----------------------------------------------------------------------
        # VOLUME
        # ----------------------------------------------------------------------
        disp_vol = Show(reader, self.view)
        disp_vol.Representation = "Volume"
        ColorBy(disp_vol, ("POINTS", self.config['vector_name'], "Magnitude"))
        
        vel_info = reader.PointData.GetArray(self.config['vector_name'])
        max_vel = vel_info.GetRange(-1)[1] if vel_info else 1.0

        self.pwf = GetOpacityTransferFunction(self.config['vector_name'])
        self.pwf.Points = [0.0, 0.0, 0.5, 0.0, max_vel, 1.0, 0.5, 0.0]

        disp_vol.Specular = 1.0
        disp_vol.SpecularPower = 50.0
        disp_vol.OSPRayMaterial = "None"

        # ----------------------------------------------------------------------
        # CAMERA SETUP
        # ----------------------------------------------------------------------
        self.view.CameraPosition = override_camera_pos
        self.view.CameraViewUp = override_view_up
        self.view.CameraFocalPoint = self.config['focal_point']
        self.view.CameraViewAngle = self.config['camera_view_angle']
        self.view.CameraParallelScale = self.config['camera_parallel_scale']
        self.view.Update()
        
        # ----------------------------------------------------------------------
        # RENDER THE 3 PANELS
        # ----------------------------------------------------------------------
        out_dir = self.config['out_dir']
        temp_1 = os.path.join(out_dir, f"temp_{frame_idx}_1.png")
        temp_2 = os.path.join(out_dir, f"temp_{frame_idx}_2.png")
        temp_3 = os.path.join(out_dir, f"temp_{frame_idx}_3.png")
        final_out = os.path.join(out_dir, output_filename)
        
        # 1. Solid Only
        for d in solid_displays: d.Visibility = 1
        disp_wire.Visibility = 1
        disp_stream.Visibility, disp_vol.Visibility = 0, 0
        disp_stream.SetScalarBarVisibility(self.view, False)
        SaveScreenshot(temp_1, self.view, ImageResolution=self.config['resolution'], TransparentBackground=0, OverrideColorPalette='WhiteBackground')
        
        # 2. Solid + Streamlines
        disp_wire.Visibility = 1
        disp_stream.Visibility, disp_vol.Visibility = 1, 0
        disp_stream.SetScalarBarVisibility(self.view, False)
        SaveScreenshot(temp_2, self.view, ImageResolution=self.config['resolution'], TransparentBackground=0, OverrideColorPalette='WhiteBackground')
        
        # 3. Volume + Streamlines
        for d in solid_displays: d.Visibility = 0
        disp_wire.Visibility = 0
        disp_stream.Visibility, disp_vol.Visibility = 1, 1
        disp_stream.SetScalarBarVisibility(self.view, False)
        SaveScreenshot(temp_3, self.view, ImageResolution=self.config['resolution'], TransparentBackground=0, OverrideColorPalette='WhiteBackground')

        # Combine with Matplotlib
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor("white")
        titles = ["Solid Only", "Streamlines in Crystal Solid", "Fluid Volume within Streamlines"]

        for ax, img_path, title in zip(axes, [temp_1, temp_2, temp_3], titles):
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.set_title(title, fontsize=18, fontname="serif", pad=10)
            ax.axis("off")

        plt.tight_layout(pad=0.5)
        plt.savefig(final_out, dpi=120, bbox_inches="tight", pad_inches=0.05, facecolor="white", edgecolor="white")
        plt.close(fig)
        
        # Cleanup Temps
        for t in [temp_1, temp_2, temp_3]:
            try: os.remove(t)
            except: pass
        
        # Full memory recovery
        for obj in reversed(objects_to_delete):
            try: Delete(obj)
            except: pass
        for disp in solid_displays + [disp_wire, disp_stream, disp_vol]:
            try: Delete(disp)
            except: pass
            
        try: Delete(self.lut)
        except: pass
        try: Delete(self.pwf)
        except: pass
        try: Delete(self.view)
        except: pass
        
        Disconnect()
        gc.collect()


# ==============================================================================
# 3. MODULAR CAMERA SYSTEM
# ==============================================================================
def path_static(config, total_frames):
    return [config['camera_pos']] * total_frames, [config['camera_view_up']] * total_frames

def path_circular(config, total_frames):
    start_pos = np.array(config['camera_pos'])
    focal_point = np.array(config['focal_point'])
    start_up = np.array(config['camera_view_up'])
    
    r = start_pos - focal_point
    angles = np.linspace(0, np.deg2rad(config.get('rotation_angle', 360)), total_frames, endpoint=False)
    axis = config.get('rotation_axis', 'z').lower()
    
    positions, up_vectors = [], []
    for theta in angles:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        if axis == 'z':
            rx = r[0] * cos_t - r[1] * sin_t
            ry = r[0] * sin_t + r[1] * cos_t
            rz = r[2]
            
            ux = start_up[0] * cos_t - start_up[1] * sin_t
            uy = start_up[0] * sin_t + start_up[1] * cos_t
            uz = start_up[2]
            
        elif axis == 'y':
            rx = r[0] * cos_t + r[2] * sin_t
            ry = r[1]
            rz = -r[0] * sin_t + r[2] * cos_t
            
            ux = start_up[0] * cos_t + start_up[2] * sin_t
            uy = start_up[1]
            uz = -start_up[0] * sin_t + start_up[2] * cos_t
            
        elif axis == 'x':
            rx = r[0]
            ry = r[1] * cos_t - r[2] * sin_t
            rz = r[1] * sin_t + r[2] * cos_t
            
            ux = start_up[0]
            uy = start_up[1] * cos_t - start_up[2] * sin_t
            uz = start_up[1] * sin_t + start_up[2] * cos_t
            
        else:
            raise ValueError("rotation_axis in config must be 'x', 'y', or 'z'")
            
        positions.append((focal_point + np.array([rx, ry, rz])).tolist())
        up_vectors.append(np.array([ux, uy, uz]).tolist())
        
    return positions, up_vectors

def generate_camera_trajectory(config, total_frames):
    path_type = config.get('camera_path_type', 'static')
    if path_type == 'static':   return path_static(config, total_frames)
    if path_type == 'circular': return path_circular(config, total_frames)
    raise ValueError("Unknown path type.")


# ==============================================================================
# 4. CONFIGURATION
# ==============================================================================
config = {
    'preview_only':     False,
    
    'samples':          4,
    'ambient_samples':  6,
    
    'inp_dir':          './Example_Bentheimer/',
    'out_dir':          './output_frames/',
    
    'resolution':       [1200, 1200],  # Resolution of EACH panel
    'scalar_name':      'Density',     # Field for Rock/Fluid distinction
    'vector_name':      'Velocity',    # Field for Streamlines
    
    'frames':           360, 
    'camera_path_type': 'circular', 
    'rotation_angle':   360, 
    'rotation_axis':    'y',            
    
    'camera_pos':       [-145.62, 258.12, 337.27],
    'focal_point':      [54.23, 49.59, 63.19],
    'camera_view_up':   [0.28, 0.85, -0.44],
    'camera_view_angle': 30,
    'camera_parallel_scale': 103.05,
    
    'threads': 10,
}

# ==============================================================================
# 5. SUBPROCESS LAUNCHER
# ==============================================================================
def launch_worker_process(job_file_path):
    job_name = os.path.basename(job_file_path)
    print(f"[Orchestrator] Launching {job_name}...", flush=True)

    clean_env = os.environ.copy()
    mpi_prefixes = ['OMPI_', 'PMIX_', 'ORTE_', 'MPI_', 'OPAL_']
    for key in list(clean_env.keys()):
        if any(key.startswith(prefix) for prefix in mpi_prefixes):
            del clean_env[key]
            
    clean_env['HWLOC_HIDE_ERRORS'] = '1'
    clean_env['OMP_NUM_THREADS'] = '1'
    clean_env['TBB_NUM_THREADS'] = '1'
    clean_env['VTK_SMP_MAX_THREADS'] = '1'
    clean_env['OSPRAY_THREADS'] = '1'
    clean_env['PYTHONUNBUFFERED'] = '1'

    cmd = ["pvpython", "--force-offscreen-rendering", __file__, "--worker", job_file_path]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=clean_env, timeout=3600)
        print(f"\n{'='*60}\n[Finished {job_name} | RC: {result.returncode}]\n{result.stdout.strip()}\n{'='*60}\n", flush=True)
        if result.returncode != 0:
            print(f"[FATAL ERROR] {job_name} failed to render.", flush=True)
    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT ERROR] {job_name} hung for 3600s.", flush=True)

    return job_name

# ==============================================================================
# 6. MAIN EXECUTION ROUTING
# ==============================================================================
if __name__ == '__main__':
    
    # ---------------------------------------------------------
    # A. WORKER MODE
    # ---------------------------------------------------------
    if '--worker' in sys.argv:
        job_file = sys.argv[sys.argv.index('--worker') + 1]
        with open(job_file, 'r') as f:
            job_data = json.load(f)
            
        print(f"--> [Worker PID: {os.getpid()}] Processing Frame {job_data['frame_idx']:03d} | Source: {os.path.basename(job_data['file_path'])}", flush=True)
        
        try:
            renderer = ParaViewRenderer(job_data['config'])
            renderer.render_file(job_data['file_path'], job_data['output_filename'], job_data['frame_idx'], job_data['camera_pos'], job_data['camera_view_up'])
            del renderer
            gc.collect()
        except Exception as e:
            print(f"--> [Worker PID: {os.getpid()}] ERROR: {e}", flush=True)
            sys.exit(1)
        
        sys.exit(0)
        
    # ---------------------------------------------------------
    # B. PREVIEW MODE (Optional, stripped down)
    # ---------------------------------------------------------
    elif '--preview' in sys.argv:
        print("Preview mode triggered. Exiting for this implementation.", flush=True)
        sys.exit(0)
        
    # ---------------------------------------------------------
    # C. ORCHESTRATOR MODE
    # ---------------------------------------------------------
    else:
        if not shutil.which("pvpython"):
            print("\n[FATAL ERROR] pvpython not found in PATH!", flush=True)
            sys.exit(1)

        os.makedirs(config['out_dir'], exist_ok=True)
        jobs_dir = os.path.join(config['out_dir'], 'job_configs')
        os.makedirs(jobs_dir, exist_ok=True)

        # Detect files (Supports both .pvti sequence OR single .vti file repeated)
        available_files = [os.path.join(r, f) for r, _, fs in os.walk(config['inp_dir']) for f in fs if f.endswith(('.pvti', '.vti'))]
        available_files.sort(key=lambda s: [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)])

        if not available_files:
            raise ValueError(f"No .pvti or .vti files found in {config['inp_dir']}")

        # If only 1 file is found, repeat it for all frames (for static camera rotations over a single object)
        if len(available_files) == 1:
            mapped_files = [available_files[0]] * config['frames']
        else:
            n_frames = len(available_files) if config['frames'] is None else config['frames']
            mapped_indices = np.round(np.linspace(0, len(available_files) - 1, n_frames)).astype(int)
            mapped_files = [available_files[i] for i in mapped_indices]
            
        pos, up = generate_camera_trajectory(config, config['frames'])

        num_workers = min(config['threads'], os.cpu_count() or 1)
        job_paths = []
        skipped_frames = 0
        
        for idx in range(config['frames']):
            output_filename = f"frame_{idx:04d}.png"
            expected_output_path = os.path.join(config['out_dir'], output_filename)
            
            if os.path.exists(expected_output_path):
                skipped_frames += 1
                continue
            
            job_data = {
                'frame_idx': idx,
                'file_path': mapped_files[idx],
                'output_filename': output_filename,
                'camera_pos': pos[idx],
                'camera_view_up': up[idx],
                'config': config 
            }
            job_path = os.path.join(jobs_dir, f"job_{idx:04d}.json")
            with open(job_path, 'w') as f:
                json.dump(job_data, f)
            job_paths.append(job_path)

        if skipped_frames > 0:
            print(f"\n[Orchestrator] Skipped {skipped_frames} existing frames.", flush=True)

        if job_paths:
            print(f"\n[Orchestrator] Initiating Subprocess Pool with {num_workers} isolated processes...", flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(launch_worker_process, path): path for path in job_paths}
                for future in concurrent.futures.as_completed(futures):
                    pass
            print("\n[Orchestrator] Render Complete.", flush=True)

        # ----------------------------------------------------------------------
        # AUTOMATIC FFmpeg VIDEO CREATION
        # ----------------------------------------------------------------------
        if shutil.which("ffmpeg") is not None:
            output_video = os.path.join(config['out_dir'], "rotating_views.mp4")
            cmd = [
                "ffmpeg", "-y", "-framerate", "30",
                "-i", os.path.join(config['out_dir'], "frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                "-preset", "medium", "-movflags", "+faststart", output_video,
            ]
            print("\n[Orchestrator] Creating video via FFmpeg...", flush=True)
            subprocess.run(cmd, check=True)
            print(f"Video saved: {output_video}")
        else:
            print("\n[Orchestrator] ffmpeg not found in PATH. Skipping video creation.")