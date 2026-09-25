import os
import sys
import gc
import json
import subprocess
import shutil
import concurrent.futures
import numpy as np
import copy
import colorsys

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
# 2. UTILITIES
# ==============================================================================
def cuda_available():
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        subprocess.check_output(
            ["nvidia-smi", "-L"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5
        )
        return True
    except Exception:
        return False

def get_files_dict(directory):
    files_dict = {}
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")
        
    for f in os.listdir(directory):
        if f.startswith("train_checkpoint_"):
            full_path = os.path.join(directory, f)
            if os.path.isfile(full_path):
                name_no_ext = os.path.splitext(f)[0]
                try:
                    number_part = name_no_ext.split("train_checkpoint_")[-1]
                    files_dict[int(number_part)] = f
                except ValueError:
                    continue
                    
    if not files_dict:
        raise ValueError(f"No valid checkpoint files found in {directory}.")
    return dict(sorted(files_dict.items()))

def safe_get_epochs(directory):
    try:
        return get_files_dict(directory)
    except Exception:
        return {}

# ==============================================================================
# 3. IMMEDIATE-MODE RENDERER
# ==============================================================================
class ParaViewRenderer:
    def __init__(self, config):
        self.config = config
        self.view = None
        self.lut = None
        self.pwf = None

        self.wireframe_lines = []
        self.wireframe_tubes = []
        self.wireframe_displays = []
        
        self.filter_lines = []
        self.filter_tubes = []

    def _setup_view(self):
        for v in GetRenderViews():
            try: Delete(v)
            except: pass
        gc.collect()

        self.view = CreateRenderView()
        self.view.ViewSize = self.config['resolution']
        if hasattr(self.view, 'UseColorPaletteForBackground'): self.view.UseColorPaletteForBackground = 0
        self.view.Background = [1.0, 1.0, 1.0]
        self.view.Background2 = [1.0, 1.0, 1.0]
        self.view.OrientationAxesVisibility = 0

        self.view.EnableRayTracing = 1
        self.view.SamplesPerPixel = self.config.get('samples', 4)
        self.view.AmbientSamples = self.config.get('ambient_samples', 6)

        backend = "OSPRay pathtracer"
        try:
            if "OptiX pathtracer" in list(self.view.GetProperty("BackEnd").GetAvailable()) and cuda_available():
                backend = "OptiX pathtracer"
        except: pass
        self.view.BackEnd = backend

        if hasattr(self.view, 'Backgroundmode'): self.view.Backgroundmode = 'Backplate'
        if hasattr(self.view, 'EnvironmentalBG'): self.view.EnvironmentalBG = [0.6, 0.5803, 0.8]
        if hasattr(self.view, 'UseEnvironmentLighting'): self.view.UseEnvironmentLighting = 1
        if hasattr(self.view, 'EnvironmentNorth'): self.view.EnvironmentNorth = [0.0, 1.0, 0.0]
        if hasattr(self.view, 'EnvironmentEast'): self.view.EnvironmentEast = [1.0, 0.0, 0.0]
        
        if hasattr(self.view, 'RouletteDepth'): self.view.RouletteDepth = 5
        self.view.Shadows = 1
        if hasattr(self.view, 'ProgressivePasses'): self.view.ProgressivePasses = 3
        if hasattr(self.view, 'Denoise'): self.view.Denoise = 1
        if hasattr(self.view, 'UseLight'): self.view.UseLight = 1

    def _create_custom_wireframe(self, bounds):
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        xc, yc, zc = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)

        wireframe_coords = [
            ([xc, yc, zc], [xmin, yc, zc]), ([xc, yc, zc], [xc, ymax, zc]), ([xc, yc, zc], [xc, yc, zmax]),
            ([xmin, yc, zc], [xmin, ymax, zc]), ([xmin, yc, zc], [xmin, yc, zmax]),
            ([xc, ymax, zc], [xmin, ymax, zc]), ([xc, ymax, zc], [xc, ymax, zmax]),
            ([xc, yc, zmax], [xmin, yc, zmax]), ([xc, yc, zmax], [xc, ymax, zmax]),
            ([xmin, ymin, zmin], [xmax, ymin, zmin]), ([xmin, ymax, zmin], [xmax, ymax, zmin]),
            ([xmin, ymin, zmax], [xmax, ymin, zmax]), ([xmin, ymin, zmin], [xmin, ymax, zmin]),
            ([xmax, ymin, zmin], [xmax, ymax, zmin]), ([xmax, ymin, zmax], [xmax, ymax, zmax]),
            ([xmin, ymin, zmin], [xmin, ymin, zmax]), ([xmax, ymin, zmin], [xmax, ymin, zmax]),
            ([xmax, ymax, zmin], [xmax, ymax, zmax]), ([xmin, ymax, zmin], [xmin, ymax, zc]),
            ([xmax, ymax, zmax], [xc, ymax, zmax]), ([xmin, ymin, zmax], [xmin, yc, zmax])
        ]

        for pt1, pt2 in wireframe_coords:
            line = Line(Point1=pt1, Point2=pt2)
            tube = Tube(Input=line)
            tube.Radius = self.config.get('wireframe_radius', 0.3)
            tube.Capping = 1
            disp = Show(tube, self.view)
            disp.ColorArrayName = ['POINTS', '']
            disp.DiffuseColor, disp.AmbientColor, disp.Specular = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0
            self.wireframe_lines.append(line)
            self.wireframe_tubes.append(tube)
            self.wireframe_displays.append(disp)

    def _create_conv_filter_box(self, bounds, progress):
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        size = self.config.get('filter_size', 40.0)
        
        # Ensure domain is properly bounded to prevent filter from exiting bounds
        avail_x = max(0, (xmax - xmin) - size)
        avail_y = max(0, (ymax - ymin) - size)
        avail_z = max(0, (zmax - zmin) - size)
        
        # Triangle wave generates a continuous, sweeping raster scan mapping
        def triangle(t): return 1.0 - abs((t % 2.0) - 1.0)
        
        # Z advances slowly, Y sweeps moderately, X sweeps fast
        tz = progress
        ty = triangle(progress * 3.0)
        tx = triangle(progress * 9.0)
        
        x0 = xmin + tx * avail_x
        y0 = ymin + ty * avail_y
        z0 = zmin + tz * avail_z
        x1, y1, z1 = x0 + size, y0 + size, z0 + size
        
        edges = [
            # Internal 3D Box
            ([x0, y0, z0], [x1, y0, z0]), ([x1, y0, z0], [x1, y1, z0]), 
            ([x1, y1, z0], [x0, y1, z0]), ([x0, y1, z0], [x0, y0, z0]),
            ([x0, y0, z1], [x1, y0, z1]), ([x1, y0, z1], [x1, y1, z1]), 
            ([x1, y1, z1], [x0, y1, z1]), ([x0, y1, z1], [x0, y0, z1]),
            ([x0, y0, z0], [x0, y0, z1]), ([x1, y0, z0], [x1, y0, z1]), 
            ([x1, y1, z0], [x1, y1, z1]), ([x0, y1, z0], [x0, y1, z1])
        ]
        
        hue = (progress * 2.0) % 1.0 
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        dynamic_color = [r, g, b]
        
        displays = []
        for pt1, pt2 in edges:
            line = Line(Point1=pt1, Point2=pt2)
            tube = Tube(Input=line)
            tube.Radius = self.config.get('wireframe_radius', 0.3) * 1.5
            tube.Capping = 1
            disp = Show(tube, self.view)
            disp.ColorArrayName = ['POINTS', '']
            
            # Emit pure dynamic color, ignore scene shadows
            disp.AmbientColor = dynamic_color
            disp.DiffuseColor = dynamic_color
            disp.Ambient = 1.0
            disp.Diffuse = 0.0
            disp.Specular = 0.0
            disp.OSPRayMaterial = "None"
            
            self.filter_lines.append(line)
            self.filter_tubes.append(tube)
            displays.append(disp)
            
        return displays

    def render_file(self, target_path, pred_path, output_filename, epoch, override_camera_pos, override_view_up):
        self._setup_view()
        objects_to_delete = []

        reader_target = XMLImageDataReader(FileName=[target_path])
        reader_pred = XMLImageDataReader(FileName=[pred_path])
        reader_target.PointArrayStatus = [self.config['scalar_name'], self.config['vector_name']]
        reader_pred.PointArrayStatus = [self.config['scalar_name'], self.config['vector_name']]
        reader_target.UpdatePipeline()
        reader_pred.UpdatePipeline()
        objects_to_delete.extend([reader_target, reader_pred])

        bounds = reader_target.GetDataInformation().GetBounds()
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        center = [0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)]

        thresh = Threshold(Input=reader_target)
        thresh.Scalars = ['POINTS', self.config['scalar_name']]
        thresh.ThresholdMethod = 'Between'
        thresh.LowerThreshold = -1e10
        thresh.UpperThreshold = 0.0

        clip1 = Clip(Input=thresh); clip1.ClipType = "Plane"; clip1.ClipType.Normal = [-1.0, 0.0, 0.0]; clip1.ClipType.Origin = center
        clip2 = Clip(Input=thresh); clip2.ClipType = "Plane"; clip2.ClipType.Normal = [0.0, 1.0, 0.0]; clip2.ClipType.Origin = center
        clip3 = Clip(Input=thresh); clip3.ClipType = "Plane"; clip3.ClipType.Normal = [0.0, 0.0, 1.0]; clip3.ClipType.Origin = center
        objects_to_delete.extend([thresh, clip1, clip2, clip3])

        self._create_custom_wireframe(bounds)
        objects_to_delete.extend(self.wireframe_lines + self.wireframe_tubes)
        
        filter_displays = self._create_conv_filter_box(bounds, self.config.get('progress', 0.0))
        objects_to_delete.extend(self.filter_lines + self.filter_tubes)

        solid_displays = []
        for clip_obj in [clip1, clip2, clip3]:
            disp_solid = Show(clip_obj, self.view)
            disp_solid.Representation = 'Surface'
            ColorBy(disp_solid, None)
            disp_solid.AmbientColor = self.config['solid_ambient']
            disp_solid.DiffuseColor = self.config['solid_diffuse']
            disp_solid.Opacity = 1.0
            disp_solid.Specular = 0.3
            disp_solid.SpecularPower = 100.0
            disp_solid.OSPRayMaterial = "None"
            solid_displays.append(disp_solid)

        def create_flow_viz(reader_input):
            stream = StreamTracer(Input=reader_input, SeedType="Point Cloud")
            stream.Vectors = ["POINTS", self.config['vector_name']]
            stream.MaximumStreamlineLength = 1000.0
            stream.SeedType.Center = center
            stream.SeedType.Radius = (xmax - xmin)
            stream.SeedType.NumberOfPoints = 20000
            
            disp_stream = Show(stream, self.view)
            ColorBy(disp_stream, ("POINTS", self.config['vector_name'], "Magnitude"))
            disp_stream.RenderLinesAsTubes = 1
            disp_stream.LineWidth = 0.5
            disp_stream.Ambient = 1.0
            disp_stream.Diffuse = 0.0
            disp_stream.Specular = 0.0
            disp_stream.OSPRayMaterial = "None"
            
            disp_vol = Show(reader_input, self.view)
            disp_vol.Representation = "Volume"
            disp_vol.OSPRayMaterial = 'None'
            disp_vol.Opacity = 0.8
            disp_vol.Shade = 0
            disp_vol.Ambient = 1.0
            disp_vol.Diffuse = 0.0
            disp_vol.Specular = 0.0
            ColorBy(disp_vol, ("POINTS", self.config['vector_name'], "Magnitude"))
            return stream, disp_stream, disp_vol

        stream_t, disp_stream_t, disp_vol_t = create_flow_viz(reader_target)
        stream_p, disp_stream_p, disp_vol_p = create_flow_viz(reader_pred)
        objects_to_delete.extend([stream_t, stream_p])

        # ======================================================================
        # Z-AXIS DIRECTION ARROW (3D)
        # ======================================================================
        z_arrow = Arrow()
        z_arrow.TipResolution = 32
        z_arrow.ShaftResolution = 32

        domain_length = max(xmax - xmin, ymax - ymin, zmax - zmin)
        arr_scale = domain_length * 0.3

        z_transform = Transform(Input=z_arrow)
        z_transform.Transform.Rotate = [0.0, -90.0, 0.0]
        z_transform.Transform.Scale = [arr_scale, arr_scale, arr_scale]
        z_transform.Transform.Translate = [center[0], ymin - (domain_length * 0.15), center[2] - (arr_scale * 0.5)]

        disp_z_arrow = Show(z_transform, self.view)
        disp_z_arrow.AmbientColor = [0.9, 0.2, 0.2]
        disp_z_arrow.DiffuseColor = [0.9, 0.2, 0.2]
        disp_z_arrow.Ambient = 1.0 
        disp_z_arrow.Diffuse = 0.0
        disp_z_arrow.Specular = 0.0
        disp_z_arrow.OSPRayMaterial = "None"
        
        objects_to_delete.extend([z_arrow, z_transform])

        # ======================================================================
        # COLORMAP SYNC
        # ======================================================================
        vel_info = reader_target.PointData.GetArray(self.config['vector_name'])
        min_vel = vel_info.GetRange(-1)[0] if vel_info else 0.0
        max_vel = vel_info.GetRange(-1)[1] if vel_info else 1.0

        self.lut = GetColorTransferFunction(self.config['vector_name'])
        self.lut.ApplyPreset("Plasma (matplotlib)", True)
        
        pts = list(self.lut.RGBPoints)
        x_coords = pts[0::4]
        old_min, old_max = min(x_coords), max(x_coords)
        cutoff = old_min + 0.1 * (old_max - old_min)
        
        new_pts = []
        for i in range(0, len(pts), 4):
            x, r, g, b = pts[i:i+4]
            if x >= cutoff:
                norm_x = (x - cutoff) / (old_max - cutoff)
                new_x = min_vel + norm_x * (max_vel - min_vel)
                new_pts.extend([new_x, r, g, b])
        self.lut.RGBPoints = new_pts

        self.pwf = GetOpacityTransferFunction(self.config['vector_name'])
        self.pwf.Points = [
            0.0,     0.0, 0.5, 0.0,
            max_vel, 0.8, 0.5, 0.0   
        ]

        self.view.CameraPosition = override_camera_pos
        self.view.CameraViewUp = override_view_up
        self.view.CameraFocalPoint = self.config['focal_point']
        self.view.CameraViewAngle = self.config['camera_view_angle']
        self.view.CameraParallelScale = self.config['camera_parallel_scale']
        self.view.Update()

        out_dir = self.config['out_dir']
        temp_1_solid = os.path.join(out_dir, f"temp_{epoch}_1_solid.png")
        temp_1_wire  = os.path.join(out_dir, f"temp_{epoch}_1_wire.png")
        temp_2       = os.path.join(out_dir, f"temp_{epoch}_2.png")
        temp_3       = os.path.join(out_dir, f"temp_{epoch}_3.png")
        final_out    = os.path.join(out_dir, output_filename)

        # PANEL 1 BASE: SOLID GEOMETRY ONLY
        for d in solid_displays: d.Visibility = 1
        for d in self.wireframe_displays: d.Visibility = 0
        for d in filter_displays: d.Visibility = 0
        disp_stream_t.Visibility = disp_vol_t.Visibility = disp_stream_p.Visibility = disp_vol_p.Visibility = 0
        disp_z_arrow.Visibility = 0
        if hasattr(self.view, 'LightScale'): self.view.LightScale = 1.0
        SaveScreenshot(temp_1_solid, self.view, ImageResolution=self.config['resolution'], TransparentBackground=0, OverrideColorPalette='WhiteBackground')

        # PANEL 1 OVERLAY: WIREFRAMES ONLY (TRANSPARENT)
        for d in solid_displays: d.Visibility = 0
        for d in self.wireframe_displays: d.Visibility = 1
        for d in filter_displays: d.Visibility = 1
        SaveScreenshot(temp_1_wire, self.view, ImageResolution=self.config['resolution'], TransparentBackground=1)

        # PANEL 2: PREDICTION FLOW 
        for d in self.wireframe_displays: d.Visibility = 1
        for d in filter_displays: d.Visibility = 0
        disp_stream_p.Visibility = disp_vol_p.Visibility = 1
        disp_stream_p.Opacity = 1.0
        disp_z_arrow.Visibility = 1
        self.view.Shadows = 0
        if hasattr(self.view, 'LightScale'): self.view.LightScale = 1.5
        SaveScreenshot(temp_2, self.view, ImageResolution=self.config['resolution'], TransparentBackground=0, OverrideColorPalette='WhiteBackground')

        # PANEL 3: TARGET FLOW 
        disp_stream_p.Visibility = disp_vol_p.Visibility = 0
        disp_stream_t.Visibility = disp_vol_t.Visibility = 1
        disp_stream_t.Opacity = 1.0
        disp_z_arrow.Visibility = 0
        SaveScreenshot(temp_3, self.view, ImageResolution=self.config['resolution'], TransparentBackground=0, OverrideColorPalette='WhiteBackground')

        # COMBINE PANELS WITH COMPOSITING
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor("white")
        titles = ["Solid Geometry", f"Predicted Flow (Epoch {epoch})", "Target Flow"]

        axes[0].imshow(mpimg.imread(temp_1_solid))
        axes[0].imshow(mpimg.imread(temp_1_wire))
        axes[0].set_title(titles[0], fontsize=18, fontname="serif", pad=10)
        axes[0].axis("off")

        axes[1].imshow(mpimg.imread(temp_2))
        axes[1].set_title(titles[1], fontsize=18, fontname="serif", pad=10)
        axes[1].axis("off")

        axes[2].imshow(mpimg.imread(temp_3))
        axes[2].set_title(titles[2], fontsize=18, fontname="serif", pad=10)
        axes[2].axis("off")

        plt.tight_layout(pad=0.5)
        plt.savefig(final_out, dpi=120, bbox_inches="tight", pad_inches=0.05, facecolor="white", edgecolor="white")
        plt.close(fig)

        for t in [temp_1_solid, temp_1_wire, temp_2, temp_3]:
            try: os.remove(t)
            except: pass
            
        for disp in self.wireframe_displays + solid_displays + filter_displays + [disp_stream_t, disp_vol_t, disp_stream_p, disp_vol_p, disp_z_arrow]:
            try: Delete(disp)
            except: pass
        for obj in reversed(objects_to_delete):
            try: Delete(obj)
            except: pass
        for item in [self.lut, self.pwf, self.view]:
            try: Delete(item)
            except: pass
        Disconnect()
        gc.collect()

# ==============================================================================
# 4. CAMERA SYSTEM
# ==============================================================================
def path_circular(config, total_frames):
    start_pos = np.array(config['camera_pos'])
    focal_point = np.array(config['focal_point'])
    start_up = np.array(config['camera_view_up'])
    r = start_pos - focal_point
    angles = np.linspace(0, np.deg2rad(config.get('rotation_angle', 360)), total_frames, endpoint=False)
    
    positions, up_vectors = [], []
    for theta in angles:
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rx = r[0] * cos_t + r[2] * sin_t
        ry = r[1]
        rz = -r[0] * sin_t + r[2] * cos_t
        ux = start_up[0] * cos_t + start_up[2] * sin_t
        uy = start_up[1]
        uz = -start_up[0] * sin_t + start_up[2] * cos_t
        
        positions.append((focal_point + np.array([rx, ry, rz])).tolist())
        up_vectors.append(np.array([ux, uy, uz]).tolist())
    return positions, up_vectors

# ==============================================================================
# 5. CONFIGURATION
# ==============================================================================
config = {
    'samples': 4,
    'ambient_samples': 6,
    'out_dir': './output_frames_epoch/',
    'resolution': [1200, 1200],
    'scalar_name': 'Density',
    'vector_name': 'Velocity',

    'solid_ambient':  [0.6, 0.5804, 0.6706],
    'solid_diffuse':  [0.6, 0.5804, 0.6706],
    'wireframe_radius': 0.3,
    'filter_size': 40.0,
    
    'is_rotating': False,
    'rotation_angle': 60,
    'camera_pos': [-145.62, 258.12, 337.27],
    'focal_point': [54.23, 49.59, 63.19],
    'camera_view_up': [0.28, 0.85, -0.44],
    'camera_view_angle': 30,
    'camera_parallel_scale': 103.05,
    'threads': 10,
}

# ==============================================================================
# 6. SUBPROCESS LAUNCHER
# ==============================================================================
def launch_worker_process(job_file_path):
    job_name = os.path.basename(job_file_path)
    clean_env = os.environ.copy()
    for key in list(clean_env.keys()):
        if any(key.startswith(prefix) for prefix in ['OMPI_', 'PMIX_', 'ORTE_', 'MPI_', 'OPAL_']):
            del clean_env[key]
    clean_env['HWLOC_HIDE_ERRORS'] = '1'
    for k in ['OMP_NUM_THREADS', 'TBB_NUM_THREADS', 'VTK_SMP_MAX_THREADS', 'OSPRAY_THREADS']:
        clean_env[k] = '1'

    cmd = ["pvpython", "--force-offscreen-rendering", __file__, "--worker", job_file_path]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=clean_env, timeout=3600)
        if result.returncode != 0:
            print(f"[FATAL ERROR] {job_name} failed.\n{result.stdout.strip()}", flush=True)
    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] {job_name} hung.", flush=True)

# ==============================================================================
# 7. MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':

    if '--worker' in sys.argv:
        job_file = sys.argv[sys.argv.index('--worker') + 1]
        with open(job_file, 'r') as f: job_data = json.load(f)

        try:
            renderer = ParaViewRenderer(job_data['config'])
            renderer.render_file(
                job_data['target_vti'], job_data['pred_vti'], job_data['output_filename'], 
                job_data['epoch'], job_data['camera_pos'], job_data['camera_view_up']
            )
        except Exception as e:
            print(f"--> [Worker PID: {os.getpid()}] ERROR: {e}", flush=True)
            sys.exit(1)
        sys.exit(0)

    elif '--preview' in sys.argv:
        print("Preview mode triggered. Exiting.", flush=True)
        sys.exit(0)

    else:
        print("\n[Orchestrator] Initializing PyTorch Setup...", flush=True)
        import torch
        import torch.nn as nn
        import pyvista as pv
        import copy
        
        from Architectures.Unet import Extended_DannyKo
        from Utilities import dataset_reader as dr
        from Utilities import nn_trainner as nnt

        try:
            from Functional import Channel_Concat, Ux2Uy
        except ImportError:
            from Architectures.Functional import Channel_Concat, Ux2Uy

        # ======================================================================
        # COMPOSITE ARCHITECTURE CLASS
        # ======================================================================
        class SubModels_Composition(nn.Module):
            def __init__(self, main_model, bin_input=True):
                super().__init__() 
                
                for attr in ['z_model', 'x_model', 'p_model']:
                    if not (hasattr(main_model, attr) and isinstance(getattr(main_model, attr), nn.Module)): 
                        raise AttributeError(f"Provided main_model is missing required attribute: {attr}")
                
                self.z_model = copy.deepcopy(main_model.z_model)
                self.x_model = copy.deepcopy(main_model.x_model)
                self.y_model = Ux2Uy(self.x_model)
                self.p_model = copy.deepcopy(main_model.p_model)
                
                self.z_model.bin_input = bin_input
                self.x_model.bin_input = bin_input
                self.p_model.bin_input = bin_input
                
                self.concat = Channel_Concat()
                
            def forward(self, x):
                with torch.no_grad():
                    x_out = self.x_model.predict(x) if hasattr(self.x_model, 'predict') else self.x_model(x)
                    y_out = self.y_model.predict(x) if hasattr(self.y_model, 'predict') else self.y_model(x)
                    z_out = self.z_model.predict(x) if hasattr(self.z_model, 'predict') else self.z_model(x)
                    p_out = self.p_model.predict(x) if hasattr(self.p_model, 'predict') else self.p_model(x)
                        
                return self.concat(z_out, y_out, x_out, p_out)
            
            def predict(self, x):
                with torch.no_grad():
                    out  = self.forward(x)
                    mask = (x > 0).to(torch.float32) 
                    mask = mask.expand(-1, out.shape[1], -1, -1, -1)
                    return out * mask

        # ======================================================================
        # DATASET SETUP
        # ======================================================================
        device = 'cpu'
        dataset_path = "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_Berea_SAug_DNorm.h5"
        sample_index = 15

        dataset = dr.LazyDatasetTorch(h5_path=dataset_path, list_ids=None, x_dtype=torch.float32, y_dtype=torch.float32)
        if sample_index >= len(dataset):
            raise IndexError(f"Requested sample {sample_index}, but dataset only has {len(dataset)} samples.")
        
        print(f"[Orchestrator] Loading sample index {sample_index}...", flush=True)
        net_input, net_target = dataset[sample_index]
        net_input = net_input.unsqueeze(0).to(dtype=torch.float32)
        net_target = net_target.unsqueeze(0).to(dtype=torch.float32)

        os.makedirs(config['out_dir'], exist_ok=True)
        target_vti_path = os.path.join(config['out_dir'], "target_sample.vti")

        def save_to_vti(geometry_tensor, ux, uy, uz, filepath):
            geom_np = geometry_tensor.cpu().numpy()
            nz, ny, nx = geom_np.shape
            
            grid = pv.ImageData(dimensions=(nx, ny, nz))
            density = np.where(geom_np > 0, 1.0, -1.0).astype(np.float32)
            grid["Density"] = density.flatten()
            vel = np.stack((ux, uy, uz), axis=-1).astype(np.float32)
            grid["Velocity"] = vel.reshape(-1, 3)
            grid.set_active_scalars("Density")
            grid.set_active_vectors("Velocity")
            grid.save(filepath)

        geom = net_input[0, 0]
        uz_t, uy_t, ux_t = net_target[0, 0].cpu().numpy(), net_target[0, 1].cpu().numpy(), net_target[0, 2].cpu().numpy()
        save_to_vti(geom, ux_t, uy_t, uz_t, target_vti_path)

        # ======================================================================
        # MODEL IMPORT & MULTI-EPOCH DISCOVERY
        # ======================================================================
        base_model = Extended_DannyKo()
        base_model.bin_input = True 
        
        concat_model = SubModels_Composition(base_model, bin_input=True)

        z_dir = "../NN_Results/NN_Trainning_26_August_2026_03-45PM_Job27376/"
        x_dir = "../NN_Results/NN_Trainning_26_August_2026_06-21PM_Job27380/"
        p_dir = "../NN_Results/NN_Trainning_26_August_2026_03-47PM_Job27377/"
        
        files_z = safe_get_epochs(z_dir)
        files_x = safe_get_epochs(x_dir)
        files_p = safe_get_epochs(p_dir)
        
        all_epochs = set(files_z.keys()) | set(files_x.keys()) | set(files_p.keys())
        if not all_epochs:
            print("[FATAL ERROR] No checkpoints found in any of the specified directories.", flush=True)
            sys.exit(1)
            
        sorted_epochs = sorted(list(all_epochs))
        total_frames = len(sorted_epochs)
        
        z_epochs = sorted(list(files_z.keys()))
        x_epochs = sorted(list(files_x.keys()))
        p_epochs = sorted(list(files_p.keys()))

        if config.get('is_rotating', False):
            pos_list, up_list = path_circular(config, total_frames)
        else:
            pos_list = [config['camera_pos']] * total_frames
            up_list = [config['camera_view_up']] * total_frames

        jobs_dir = os.path.join(config['out_dir'], 'job_configs')
        os.makedirs(jobs_dir, exist_ok=True)

        futures = []
        num_workers = min(config['threads'], os.cpu_count() or 1)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)

        last_z, last_x, last_p = -1, -1, -1

        print("[Orchestrator] Generating Predictions and Dispatching Render Workers...", flush=True)
        for idx, current_epoch in enumerate(sorted_epochs):
            output_filename = f"frame_idx_{idx:04d}_epoch_{current_epoch:03d}.png"
            if os.path.exists(os.path.join(config['out_dir'], output_filename)): 
                continue
            
            target_z = max([e for e in z_epochs if e <= current_epoch], default=-1)
            target_x = max([e for e in x_epochs if e <= current_epoch], default=-1)
            target_p = max([e for e in p_epochs if e <= current_epoch], default=-1)

            if target_z != -1 and target_z != last_z:
                concat_model.z_model, _ = nnt.load_model_from_checkpoint(concat_model.z_model, z_dir, epoch=target_z, device=device)
                last_z = target_z
                
            if target_x != -1 and target_x != last_x:
                concat_model.x_model, _ = nnt.load_model_from_checkpoint(concat_model.x_model, x_dir, epoch=target_x, device=device)
                concat_model.y_model = Ux2Uy(concat_model.x_model)
                last_x = target_x
                
            if target_p != -1 and target_p != last_p:
                concat_model.p_model, _ = nnt.load_model_from_checkpoint(concat_model.p_model, p_dir, epoch=target_p, device=device)
                last_p = target_p
            
            concat_model.eval()
            
            with torch.no_grad():
                net_output = concat_model.predict(net_input)
            
            uz_p, uy_p, ux_p = net_output[0, 0].cpu().numpy(), net_output[0, 1].cpu().numpy(), net_output[0, 2].cpu().numpy()
            
            pred_vti_path = os.path.join(config['out_dir'], f"temp_pred_epoch_{current_epoch:03d}.vti")
            save_to_vti(geom, ux_p, uy_p, uz_p, pred_vti_path)

            config_for_job = copy.deepcopy(config)
            config_for_job['progress'] = idx / max(1, total_frames - 1)

            job_data = {
                'epoch': current_epoch,
                'target_vti': target_vti_path,
                'pred_vti': pred_vti_path,
                'output_filename': output_filename,
                'camera_pos': pos_list[idx],
                'camera_view_up': up_list[idx],
                'config': config_for_job
            }
            
            job_path = os.path.join(jobs_dir, f"job_{idx:04d}.json")
            with open(job_path, 'w') as f: json.dump(job_data, f)
            
            future = executor.submit(launch_worker_process, job_path)
            
            def make_cleanup_callback(v_path, j_path):
                def cleanup(fut):
                    try:
                        if os.path.exists(v_path): os.remove(v_path)
                        if os.path.exists(j_path): os.remove(j_path)
                    except Exception:
                        pass
                return cleanup
                
            future.add_done_callback(make_cleanup_callback(pred_vti_path, job_path))
            futures.append(future)

        if futures:
            print(f"\n[Orchestrator] Waiting for {len(futures)} rendering tasks to complete...", flush=True)
            concurrent.futures.wait(futures)
        
        executor.shutdown()
        
        if os.path.exists(target_vti_path): os.remove(target_vti_path)

        if shutil.which("ffmpeg"):
            output_video = os.path.join(config['out_dir'], "learning_process.mp4")
            cmd = [
                "ffmpeg", "-y", "-framerate", "10", 
                "-pattern_type", "glob", "-i", os.path.join(config['out_dir'], "frame_idx_*.png"),
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", 
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", "-movflags", "+faststart",
                output_video
            ]
            print("\n[Orchestrator] Creating video via FFmpeg...", flush=True)
            subprocess.run(cmd, check=True)
            print(f"Video saved: {output_video}")