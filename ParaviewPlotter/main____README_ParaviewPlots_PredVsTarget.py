import os
import sys
import gc
import json
import subprocess
import shutil
import numpy as np
import copy
import colorsys

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
        
        avail_x = max(0, (xmax - xmin) - size)
        avail_y = max(0, (ymax - ymin) - size)
        avail_z = max(0, (zmax - zmin) - size)
        
        def triangle(t): return 1.0 - abs((t % 2.0) - 1.0)
        
        tz = progress
        ty = triangle(progress * 3.0)
        tx = triangle(progress * 9.0)
        
        x0 = xmin + tx * avail_x
        y0 = ymin + ty * avail_y
        z0 = zmin + tz * avail_z
        x1, y1, z1 = x0 + size, y0 + size, z0 + size
        
        edges = [
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

    def render_file(self, target_path, pred_path, output_filename, override_camera_pos, override_view_up):
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
        
        filter_displays = self._create_conv_filter_box(bounds, self.config.get('progress', 1.0))
        objects_to_delete.extend(self.filter_lines + self.filter_tubes)

        solid_displays = []
        for clip_obj in [clip1, clip2, clip3]:
            disp_solid = Show(clip_obj, self.view)
            disp_solid.Representation = 'Surface'
            ColorBy(disp_solid, None)
            disp_solid.AmbientColor = self.config['solid_ambient']
            disp_solid.DiffuseColor = self.config['solid_diffuse']
            
            # Specular constraints
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

        # ======================================================================
        # EXPORTS (SEPARATE IMAGES)
        # ======================================================================
        out_dir = self.config['out_dir']
        base_name = output_filename.replace('.png', '')
        
        img1_path = os.path.join(out_dir, f"{base_name}_1_Solid.png")
        img2_path = os.path.join(out_dir, f"{base_name}_2_Solid_Flow.png")
        img3_path = os.path.join(out_dir, f"{base_name}_3_Flow.png")

        # IMAGE 1: SOLID GEOMETRY ONLY
        for d in solid_displays: 
            d.Visibility = 1
            d.Opacity = 1.0  # Fully opaque
            
        for d in self.wireframe_displays: d.Visibility = 1
        for d in filter_displays: d.Visibility = 1
        disp_z_arrow.Visibility = 1
        
        # Hide all flows
        disp_stream_t.Visibility = disp_vol_t.Visibility = 0
        disp_stream_p.Visibility = disp_vol_p.Visibility = 0
        
        if hasattr(self.view, 'LightScale'): self.view.LightScale = 1.0
        SaveScreenshot(img1_path, self.view, ImageResolution=self.config['resolution'], TransparentBackground=0, OverrideColorPalette='WhiteBackground')

        # IMAGE 2: LESS OPAQUE GEOMETRY + STREAMLINES + VOID VOLUME (Prediction)
        for d in solid_displays: 
            d.Visibility = 1
            d.Opacity = 0.25  # Reduced opacity geometry
            
        # Enable Prediction Flow
        disp_stream_p.Visibility = disp_vol_p.Visibility = 1
        disp_stream_p.Opacity = 1.0
        
        self.view.Shadows = 0 # Turn off shadows to prevent blackening of the transparent solid
        if hasattr(self.view, 'LightScale'): self.view.LightScale = 1.5
        SaveScreenshot(img2_path, self.view, ImageResolution=self.config['resolution'], TransparentBackground=0, OverrideColorPalette='WhiteBackground')

        # IMAGE 3: STREAMLINES + VOID VOLUME (Prediction only)
        for d in solid_displays: 
            d.Visibility = 0  # Hide solid
            
        # Arrow, wireframes, and flow remain visible
        SaveScreenshot(img3_path, self.view, ImageResolution=self.config['resolution'], TransparentBackground=0, OverrideColorPalette='WhiteBackground')

        # ======================================================================
        # CLEANUP
        # ======================================================================
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
# 4. CONFIGURATION
# ==============================================================================
config = {
    'samples': 4,
    'ambient_samples': 6,
    'out_dir': './output_single_frame/',
    'resolution': [1200, 1200],
    'scalar_name': 'Density',
    'vector_name': 'Velocity',

    'solid_ambient':  [0.6, 0.5804, 0.6706],
    'solid_diffuse':  [0.6, 0.5804, 0.6706],
    'wireframe_radius': 0.3,
    'filter_size': 40.0,
    
    'camera_pos': [-145.62, 258.12, 337.27],
    'focal_point': [54.23, 49.59, 63.19],
    'camera_view_up': [0.28, 0.85, -0.44],
    'camera_view_angle': 30,
    'camera_parallel_scale': 103.05,
}

# ==============================================================================
# 5. SUBPROCESS LAUNCHER
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
# 6. MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':

    if '--worker' in sys.argv:
        job_file = sys.argv[sys.argv.index('--worker') + 1]
        with open(job_file, 'r') as f: job_data = json.load(f)

        try:
            renderer = ParaViewRenderer(job_data['config'])
            renderer.render_file(
                job_data['target_vti'], job_data['pred_vti'], job_data['output_filename'], 
                job_data['camera_pos'], job_data['camera_view_up']
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
        import pyvista as pv
        
        from Architectures.Unet import Extended_DannyKo
        from Architectures.Models import SubModels_Composition
        from Utilities import dataset_reader as dr
        from Utilities import velocity_usage as vu

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # ======================================================================
        # 1. LOAD MODEL
        # ======================================================================
        danny_model = Extended_DannyKo()
        danny_model.bin_input = True 
        
        model_full_z_name = "./Trained_Models/NN_Trainning_26_August_2026_03-45PM_Job27376/model_LowerValidationLoss.pth"
        model_full_x_name = "./Trained_Models/NN_Trainning_26_August_2026_06-21PM_Job27380/model_LowerValidationLoss.pth"
        model_full_p_name = "./Trained_Models/NN_Trainning_26_August_2026_03-47PM_Job27377/model_LowerValidationLoss.pth"

        concat_model = SubModels_Composition(
            main_model=danny_model, z_name=model_full_z_name, x_name=model_full_x_name, 
            p_name=model_full_p_name, device=device, is_eval=True
        )
        concat_model.eval()

        # ======================================================================
        # 2. DATASET SETUP
        # ======================================================================
        dataset_path = "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_Berea_SAug_DNorm.h5"
        sample_index = 15

        dataset = dr.LazyDatasetTorch(h5_path=dataset_path, list_ids=None, x_dtype=torch.float32, y_dtype=torch.float32)
        if sample_index >= len(dataset):
            raise IndexError(f"Requested sample {sample_index}, but dataset only has {len(dataset)} samples.")
        
        print(f"[Orchestrator] Loading sample index {sample_index}...", flush=True)
        net_input, net_target = dataset[sample_index]
        net_input = net_input.unsqueeze(0).to(device=device, dtype=torch.float32)
        net_target = net_target.unsqueeze(0).to(device=device, dtype=torch.float32)

        os.makedirs(config['out_dir'], exist_ok=True)
        target_vti_path = os.path.join(config['out_dir'], "target_sample.vti")
        pred_vti_path = os.path.join(config['out_dir'], "pred_sample.vti")

        def save_to_vti(geometry_tensor, ux, uy, uz, filepath):
            geom_np = geometry_tensor.cpu().numpy()
            nz, ny, nx = geom_np.shape
            
            grid = pv.ImageData(dimensions=(nx, ny, nz))
            density = np.where(geom_np > 0, 1.0, -1.0).astype(np.float32)
            grid["Density"] = density.flatten(order="C")
            
            vel = np.column_stack((
                ux.flatten(order="C"), 
                uy.flatten(order="C"), 
                uz.flatten(order="C")
            )).astype(np.float32)
            
            grid["Velocity"] = vel
            grid.set_active_scalars("Density")
            grid.set_active_vectors("Velocity")
            grid.save(filepath)

        # ======================================================================
        # 3. PREDICT & EXPORT VTI
        # ======================================================================
        with torch.no_grad():
            pred = concat_model.predict(net_input)
            pred = vu.tensor_denorm(out=pred, inp=net_input)

        geom = net_input[0, 0]
        uz_t, uy_t, ux_t = net_target[0, 0].cpu().numpy(), net_target[0, 1].cpu().numpy(), net_target[0, 2].cpu().numpy()
        uz_p, uy_p, ux_p = pred[0, 0].cpu().numpy(), pred[0, 1].cpu().numpy(), pred[0, 2].cpu().numpy()
        
        save_to_vti(geom, ux_t, uy_t, uz_t, target_vti_path)
        save_to_vti(geom, ux_p, uy_p, uz_p, pred_vti_path)

        # ======================================================================
        # 4. DISPATCH SINGLE RENDER JOB
        # ======================================================================
        jobs_dir = os.path.join(config['out_dir'], 'job_configs')
        os.makedirs(jobs_dir, exist_ok=True)

        job_path = os.path.join(jobs_dir, "single_render_job.json")
        output_filename = "static_prediction.png"
        
        config_for_job = copy.deepcopy(config)
        config_for_job['progress'] = 1.0 # Filter box static at the end of domain

        job_data = {
            'target_vti': target_vti_path,
            'pred_vti': pred_vti_path,
            'output_filename': output_filename,
            'camera_pos': config['camera_pos'],
            'camera_view_up': config['camera_view_up'],
            'config': config_for_job
        }
        
        with open(job_path, 'w') as f: 
            json.dump(job_data, f)
        
        print(f"\n[Orchestrator] Launching render worker...", flush=True)
        launch_worker_process(job_path)
        
        print(f"\n[Orchestrator] Render complete! Files saved to {config['out_dir']}")
        
        # Cleanup temporary files
        try:
            if os.path.exists(target_vti_path): os.remove(target_vti_path)
            if os.path.exists(pred_vti_path): os.remove(pred_vti_path)
            if os.path.exists(job_path): os.remove(job_path)
        except Exception:
            pass