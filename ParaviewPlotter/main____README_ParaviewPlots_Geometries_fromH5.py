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

    def render_file(self, geom_vti_path, output_filepath, override_camera_pos, override_view_up):
        self._setup_view()
        objects_to_delete = []

        reader = XMLImageDataReader(FileName=[geom_vti_path])
        reader.PointArrayStatus = [self.config['scalar_name']]
        reader.UpdatePipeline()
        objects_to_delete.append(reader)

        bounds = reader.GetDataInformation().GetBounds()
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        center = [0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)]

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

        # Camera positioning
        self.view.CameraPosition = override_camera_pos
        self.view.CameraViewUp = override_view_up
        self.view.CameraFocalPoint = self.config['focal_point']
        self.view.CameraViewAngle = self.config['camera_view_angle']
        self.view.CameraParallelScale = self.config['camera_parallel_scale']
        self.view.Update()

        if hasattr(self.view, 'LightScale'): 
            self.view.LightScale = 1.0
            
        # ----------------------------------------------------------------------
        # Render high-res screenshot and convert to PDF via PIL
        # ----------------------------------------------------------------------
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

        # Cleanup
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
    
    'camera_pos': [-145.62, 258.12, 337.27],
    'focal_point': [54.23, 49.59, 63.19],
    'camera_view_up': [0.28, 0.85, -0.44],
    'camera_view_angle': 30,
    'camera_parallel_scale': 103.05,
}

# None -> plot all samples
# [0, 5, 10] -> plot only samples 0, 5 and 10
sample_indexes = None

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
            job_data['camera_pos'], 
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
    import torch
    import pyvista as pv
    from Utilities import dataset_reader as dr

    # ======================================================================
    # DATASETS MAPPING & ROOT PLOTS DIRECTORY SETUP
    # ======================================================================
    dataset_folder = "../NN_Datasets_Grad_Dist_40_5_55/"
    datasets = {
        #"Spherical Pores":   dataset_folder + "Test_Silveira_SphPore_SAug_DNorm.h5",
        #"Spherical Grains":  dataset_folder + "Test_Silveira_SphGrain_SAug_DNorm.h5",
        
        #"Cylindrical Pores":   dataset_folder + "Test_Silveira_CylinPore_SAug_DNorm.h5",
        #"Cylindrical Grains":  dataset_folder + "Test_Silveira_CylinGrain_SAug_DNorm.h5",
        
        #"Leopard":           dataset_folder + "Test_Oliveira_Leopard_SAug_DNorm.h5",
        #"Castle Gate":       dataset_folder + "Test_Oliveira_CastleGate_SAug_DNorm.h5",
        #"Berea Upper Gray":  dataset_folder + "Test_Oliveira_BereaUpperGray_SAug_DNorm.h5",
        #"Berea Sinter Gray": dataset_folder + "Test_Oliveira_BereaSinterGray_SAug_DNorm.h5",
        #"Berea Buff":        dataset_folder + "Test_Oliveira_BereaBuff_SAug_DNorm.h5",
        #"Berea":             dataset_folder + "Test_Oliveira_Berea_SAug_DNorm.h5",
        #"Bentheimer":        dataset_folder + "Test_Oliveira_Bentheimer_SAug_DNorm.h5",
        
        "Danny":              dataset_folder + "Train_Danny_SphPore_DAug_DNorm.h5"
    }
    
    main_plots_dir = "./Plots/Geometries/"

    def save_geometry_to_vti(geom_tensor, filepath):
        if hasattr(geom_tensor, 'cpu'):
            geom_np = geom_tensor.cpu().numpy()
        else:
            geom_np = np.array(geom_tensor)
            
        if geom_np.ndim == 4:
            geom_np = geom_np[0]
            
        nz, ny, nx = geom_np.shape
        
        grid = pv.ImageData(dimensions=(nx, ny, nz))
        density = np.where(geom_np > 0, 1.0, -1.0).astype(np.float32)
        grid["Density"] = density.flatten(order="C")
        grid.set_active_scalars("Density")
        grid.save(filepath)

    # ======================================================================
    # BATCH DATASET PROCESSING
    # ======================================================================
    for dataset_name, dataset_path in datasets.items():
        if not os.path.exists(dataset_path):
            print(f"[Warning] Dataset file not found: {dataset_path}. Skipping...", flush=True)
            continue

        # Target directory: ./Plots/Geometries/<dataset_name>/
        dataset_out_dir = os.path.join(main_plots_dir, dataset_name)
        jobs_dir = os.path.join(dataset_out_dir, 'job_configs')
        os.makedirs(jobs_dir, exist_ok=True)

        print(f"\n==================================================", flush=True)
        print(f"[Orchestrator] Processing dataset: {dataset_name}", flush=True)
        print(f"[Orchestrator] Path: {dataset_path}", flush=True)
        print(f"[Orchestrator] Output directory: {dataset_out_dir}", flush=True)
        print(f"==================================================", flush=True)

        dataset = dr.LazyDatasetTorch(h5_path=dataset_path, list_ids=None, x_dtype=torch.float32, y_dtype=torch.float32)
        total_samples = len(dataset)
        print(f"[Orchestrator] Total samples found: {total_samples}", flush=True)

        if sample_indexes is None:
            indexes_to_plot = range(total_samples)
        else:
            indexes_to_plot = sample_indexes
        
        for sample_index in indexes_to_plot:
            print(f"  -> Rendering sample {sample_index + 1}/{total_samples} to PDF...", flush=True)
            
            net_input, _ = dataset[sample_index]
            geom_tensor = net_input[0]

            temp_vti_path = os.path.join(dataset_out_dir, f"_temp_sample_{sample_index}.vti")
            output_filepath = os.path.join(dataset_out_dir, f"sample_{sample_index}_solid.pdf")
            job_path = os.path.join(jobs_dir, f"job_sample_{sample_index}.json")

            # Export temporary VTI format
            save_geometry_to_vti(geom_tensor, temp_vti_path)

            job_data = {
                'geom_vti': temp_vti_path,
                'output_filepath': output_filepath,
                'camera_pos': config['camera_pos'],
                'camera_view_up': config['camera_view_up'],
                'config': config
            }

            with open(job_path, 'w') as f:
                json.dump(job_data, f)

            # Launch worker render task
            launch_worker_process(job_path)

            # Clean up intermediate temporary files
            try:
                if os.path.exists(temp_vti_path): os.remove(temp_vti_path)
                if os.path.exists(job_path): os.remove(job_path)
            except Exception:
                pass

        # Clean up empty jobs config directory
        try:
            if os.path.exists(jobs_dir) and not os.listdir(jobs_dir):
                os.rmdir(jobs_dir)
        except Exception:
            pass

    print(f"\n[Orchestrator] All dataset renderings completed successfully!", flush=True)