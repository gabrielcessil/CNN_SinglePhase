import os
import sys
import gc
import json
import subprocess
import shutil
import numpy as np
from PIL import Image

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
# 3. SOLID GEOMETRY RENDERER
# ==============================================================================
class ParaViewSolidRenderer:
    def __init__(self, config):
        self.config = config
        self.view = None
        self.wireframe_lines = []
        self.wireframe_tubes = []
        self.wireframe_displays = []

    def _setup_view(self):
        for v in GetRenderViews():
            try: Delete(v)
            except: pass
        gc.collect()

        self.view = CreateRenderView()
        self.view.ViewSize = self.config['resolution']
        if hasattr(self.view, 'UseColorPaletteForBackground'): 
            self.view.UseColorPaletteForBackground = 0
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
        except Exception: 
            pass
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

    def render_file(self, geom_vti_path, output_filepath, override_view_up):
        self._setup_view()
        objects_to_delete = []

        reader = XMLImageDataReader(FileName=[geom_vti_path])
        reader.PointArrayStatus = [self.config['scalar_name']]
        reader.UpdatePipeline()
        objects_to_delete.append(reader)

        bounds = reader.GetDataInformation().GetBounds()
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        center = [0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)]
        domain_length = max(xmax - xmin, ymax - ymin, zmax - zmin)

        # Threshold solid domain
        thresh = Threshold(Input=reader)
        thresh.Scalars = ['POINTS', self.config['scalar_name']]
        thresh.ThresholdMethod = 'Between'
        thresh.LowerThreshold = -1e10
        thresh.UpperThreshold = 0.0

        # Cut plane slices
        clip1 = Clip(Input=thresh); clip1.ClipType = "Plane"; clip1.ClipType.Normal = [-1.0, 0.0, 0.0]; clip1.ClipType.Origin = center
        clip2 = Clip(Input=thresh); clip2.ClipType = "Plane"; clip2.ClipType.Normal = [0.0, 1.0, 0.0]; clip2.ClipType.Origin = center
        clip3 = Clip(Input=thresh); clip3.ClipType = "Plane"; clip3.ClipType.Normal = [0.0, 0.0, 1.0]; clip3.ClipType.Origin = center
        objects_to_delete.extend([thresh, clip1, clip2, clip3])

        # Bounding wireframe
        self._create_custom_wireframe(bounds)
        objects_to_delete.extend(self.wireframe_lines + self.wireframe_tubes)

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

        # 3D Z-Axis Arrow Indicator
        z_arrow = Arrow()
        z_arrow.TipResolution = 32
        z_arrow.ShaftResolution = 32
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

        # Proportional Camera Scaling (maintains original angle & relative distance)
        ref_size = self.config['reference_size']
        scale_ratio = domain_length / ref_size

        orig_pos = np.array(self.config['base_camera_pos'])
        orig_focal = np.array(self.config['base_focal_point'])
        
        # Shift camera relative to the new volume center using the exact same vector/angle proportions
        offset = orig_pos - orig_focal
        scaled_offset = offset * scale_ratio
        
        self.view.CameraPosition = (np.array(center) + scaled_offset).tolist()
        self.view.CameraFocalPoint = center
        self.view.CameraViewUp = override_view_up
        self.view.CameraViewAngle = self.config['camera_view_angle']
        self.view.CameraParallelScale = self.config['base_parallel_scale'] * scale_ratio
        self.view.Update()

        if hasattr(self.view, 'LightScale'): 
            self.view.LightScale = 1.0
            
        temp_png_path = output_filepath.replace('.pdf', '_temp.png')

        SaveScreenshot(
            temp_png_path, 
            self.view, 
            ImageResolution=self.config['resolution'], 
            TransparentBackground=0, 
            OverrideColorPalette='WhiteBackground'
        )

        image = Image.open(temp_png_path)
        image_rgb = image.convert('RGB')
        image_rgb.save(output_filepath, "PDF", resolution=300.0)

        if os.path.exists(temp_png_path):
            os.remove(temp_png_path)

        for disp in self.wireframe_displays + solid_displays + [disp_z_arrow]:
            try: Delete(disp)
            except: pass
        for obj in reversed(objects_to_delete):
            try: Delete(obj)
            except: pass
        try: Delete(self.view)
        except: pass
        Disconnect()
        gc.collect()

# ==============================================================================
# 4. CONFIGURATION
# ==============================================================================
config = {
    'samples': 4,
    'ambient_samples': 6,
    'resolution': [1200, 1200],
    'scalar_name': 'Density',

    'solid_ambient':  [0.6, 0.5804, 0.6706],
    'solid_diffuse':  [0.6, 0.5804, 0.6706],
    'wireframe_radius': 0.3,
    
    # Original exact camera values tuned for 120³ geometry
    'reference_size': 120.0,
    'base_camera_pos': [-145.62, 258.12, 337.27],
    'base_focal_point': [54.23, 49.59, 63.19],
    'base_parallel_scale': 103.05,
    'camera_view_up': [0.28, 0.85, -0.44],
    'camera_view_angle': 30,
}

#root_input_folder = "../TestSpeedUp_Simulations_OOD/IC_Estaillades_256_256_256/"

root_input_folder = "../GradSimulations_Dist_40_5_55/DONE_Train_Danny_SphPore_120_120_120/"
target_shape = (120, 120, 120)

dataset_folder_name = os.path.basename(os.path.normpath(root_input_folder))
dataset_out_dir = os.path.abspath(os.path.join("./Plots/Geometries/", dataset_folder_name))

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
if '--worker' in sys.argv:
    job_file = sys.argv[sys.argv.index('--worker') + 1]
    with open(job_file, 'r') as f: 
        job_data = json.load(f)

    try:
        renderer = ParaViewSolidRenderer(job_data['config'])
        renderer.render_file(
            job_data['geom_vti'], 
            job_data['output_filepath'], 
            job_data['camera_view_up']
        )
    except Exception as e:
        print(f"--> [Worker PID: {os.getpid()}] ERROR: {e}", flush=True)
        sys.exit(1)
    sys.exit(0)

elif '--preview' in sys.argv:
    print("Preview mode triggered. Exiting.", flush=True)
    sys.exit(0)

else:
    import pyvista as pv

    def save_raw_to_vti(raw_path, filepath, shape, dtype):
        geom_np = np.fromfile(raw_path, dtype=dtype).reshape(shape)
        nz, ny, nx = geom_np.shape
        grid = pv.ImageData(dimensions=(nx, ny, nz))
        density = np.where(geom_np > 0, 1.0, -1.0).astype(np.float32)
        grid["Density"] = density.flatten(order="C")
        grid.set_active_scalars("Density")
        grid.save(filepath)

    print(f"\n==================================================", flush=True)
    print(f"[Orchestrator] Buscando arquivos .raw nas subpastas de: {root_input_folder}", flush=True)
    print(f"==================================================", flush=True)

    try:
        subfolders = [f.path for f in os.scandir(root_input_folder) if f.is_dir()]
    except FileNotFoundError:
        subfolders = []

    for subfolder_path in subfolders:
        sample_name = os.path.basename(subfolder_path)
        
        raw_files = [f for f in os.listdir(subfolder_path) if f.endswith('.raw')]
        if not raw_files:
            continue

        for file in raw_files:
            raw_path = os.path.join(subfolder_path, file)
            
            os.makedirs(dataset_out_dir, exist_ok=True)
            jobs_dir = os.path.join(dataset_out_dir, 'job_configs')
            os.makedirs(jobs_dir, exist_ok=True)

            print(f"-> Processando amostra '{sample_name}' (Arquivo: {raw_path})", flush=True)

            temp_vti_path = os.path.join(dataset_out_dir, "_temp_geom.vti")
            output_filepath = os.path.join(dataset_out_dir, f"{sample_name}_solid.pdf")
            job_path = os.path.join(jobs_dir, "job_config.json")

            try:
                save_raw_to_vti(raw_path, temp_vti_path, target_shape, np.uint8)
            except Exception as e:
                print(f"[Error] Falha ao ler {raw_path} com shape {target_shape}: {e}", flush=True)
                continue

            job_data = {
                'geom_vti': temp_vti_path,
                'output_filepath': output_filepath,
                'camera_view_up': config['camera_view_up'],
                'config': config
            }

            with open(job_path, 'w') as f:
                json.dump(job_data, f)

            launch_worker_process(job_path)

            try:
                if os.path.exists(temp_vti_path): os.remove(temp_vti_path)
                if os.path.exists(job_path): os.remove(job_path)
            except Exception:
                pass

            try:
                if os.path.exists(jobs_dir) and not os.listdir(jobs_dir):
                    os.rmdir(jobs_dir)
            except Exception:
                pass

    print(f"\n[Orchestrator] Todas as renderizações das subpastas foram concluídas!", flush=True)